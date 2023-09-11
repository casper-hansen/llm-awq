import math
import torch
from torch import nn
from torch.nn import functional as F
import awq_inference_engine

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    xq_ = torch.view_as_complex(
        xq.float().reshape(*xq.shape[:-1], 2, -1).transpose(-2, -1).contiguous()
    )
    xk_ = torch.view_as_complex(
        xk.float().reshape(*xk.shape[:-1], 2, -1).transpose(-2, -1).contiguous()
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).transpose(-2, -1).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).transpose(-2, -1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def set_module_name(model, name, value):
    if '.' in name:
        parent_name = name.rsplit('.', 1)[0]
        child_name = name[len(parent_name) + 1:]
        parent = model.get_submodule(parent_name)
    else:
        parent_name = ''
        parent = model
        child_name = name

    setattr(parent, child_name, value)


def gen_slopes(n_heads, alibi_bias_max=8):
    _n_heads = 2 ** math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = 1.0 / torch.pow(2, m)
    if _n_heads != n_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]
    return slopes.view(1, n_heads, 1, 1)


def build_alibi_bias(
    n_heads, seq_len, full=False, alibi_bias_max=8, dtype=torch.float32
):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32).view(1, 1, 1, seq_len)
    if full:
        alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.int32).view(
            1, 1, seq_len, 1
        )
        alibi_bias = alibi_bias.abs().mul(-1)
    slopes = gen_slopes(n_heads, alibi_bias_max)
    alibi_bias = alibi_bias * slopes
    slopes = slopes.squeeze(0).squeeze(-1).squeeze(-1)
    return slopes.to(dtype=dtype), alibi_bias.to(dtype=dtype)


class QuantAttentionFused(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_layer, o_proj, dev, max_seq_len, use_alibi=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = self.hidden_size // num_heads
        self.qkv_proj = qkv_layer
        self.o_proj = o_proj
        self.start_pos = 0
        self.use_alibi = use_alibi
        self.cache_batch_size = 1

        # following fastertransformer definition
        self.cache_v = (
            torch.zeros(
                ( self.cache_batch_size, self.n_local_heads, max_seq_len, self.head_dim, )
            ).to(dev).half()
        )
        
        # 8: pack 8 fp16 in FT, if fp32 then use 4
        self.cache_k = (
            torch.zeros(
                ( self.cache_batch_size, self.n_local_heads, self.head_dim // 8, max_seq_len, 8, )
            ).to(dev).half()
        )

        if use_alibi:
            alibi_slopes, alibi_bias = build_alibi_bias(self.n_local_heads, max_seq_len)
            self.alibi_slopes = alibi_slopes.float().to(dev)
            self.alibi_bias = alibi_bias.float().to(dev)
            self.rotary_dim = 0
            self.is_neox = False
        else:
            self.freqs_cis = precompute_freqs_cis(
                hidden_size // num_heads,
                max_seq_len * 2,
            ).to(dev)
            self.rotary_dim = self.head_dim
            self.alibi_slopes = None
            self.is_neox = True
    
    def _multi_query_attention_torch(self, query, key, value, batch_size, seqlen, use_cache, past_key_value, attention_mask):
        query = query.view(batch_size, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)

        if use_cache:
            key = key.contiguous()
            value = value.contiguous()
            query = query.contiguous()

        output = F.scaled_dot_product_attention(query, key, value, is_causal=past_key_value is None, attn_mask=attention_mask)

        del query, key, value

        output = output.transpose(1, 2).reshape(batch_size, seqlen, self.hidden_size)

        return output
    
    def forward(
        self,
        hidden_states, past_key_value=None, attention_mask=None, position_ids=None, output_attentions=False, use_cache=False
    ):
        bsz, seqlen, _ = hidden_states.shape
        xqkv = self.qkv_proj(hidden_states)
        xqkv = xqkv.view(bsz, seqlen, -1, self.n_local_heads, self.head_dim)
        xq = xqkv[:, :, 0]
        xk = xqkv[:, :, 1]
        xv = xqkv[:, :, 2]

        if seqlen > 1:
            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

            if not self.use_alibi:
                xq, xk = apply_rotary_emb(xq, xk, freqs_cis=self.freqs_cis[self.start_pos : self.start_pos + seqlen])

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            values_store = xv.transpose(2, 1)
            keys_store = (
                xk.reshape(bsz, seqlen, self.n_local_heads, self.head_dim // 8, 8)
                .permute(0, 2, 3, 1, 4)
                .contiguous()
            )

            self.cache_v[:bsz, :, self.start_pos : self.start_pos + seqlen, :] = values_store
            self.cache_k[:bsz, :, :, self.start_pos : self.start_pos + seqlen, :] = keys_store

            past_key_value = (xk, xv) if use_cache else None
            output = self._multi_query_attention_torch(xq, xk, xv, bsz, seqlen, True, past_key_value, attention_mask)
        else:
            xq = xq[:, 0, :, :]
            xk = xk[:, 0, :, :]
            xv = xv[:, 0, :, :]
            past_key_value = (xk, xv) if use_cache else None
            output = awq_inference_engine.single_query_attention(
                xq, # query
                xk, # key
                xv, # value
                self.cache_k, # key cache
                self.cache_v, # value cache
                None, # length per sample
                self.alibi_slopes, # alibi slopes
                self.start_pos, # timestep
                self.rotary_dim, # rotary embedding dimension
                10000, # rotary embedding base
                self.is_neox, # is neox
            )
            output = output.reshape(bsz, 1, -1)
        
        attn_output = self.o_proj(output)
        
        if use_cache:
            self.start_pos += seqlen
        else:
            self.start_pos = 0

        return attn_output, None, past_key_value

from awq.quantize.qmodule import WQLinear

class MptBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, qkv_layer: WQLinear, o_proj: WQLinear, mpt_mlp):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.attn = QuantAttentionFused(hidden_size, self.n_heads, qkv_layer, o_proj, dev="cuda:0", max_seq_len=8096, use_alibi=True).to("cuda:0")
        self.ffn = mpt_mlp.to("cuda:0")
        self.norm_1 = nn.LayerNorm(hidden_size, eps=1e-6).to("cuda:0")
        self.norm_2 = nn.LayerNorm(hidden_size, eps=1e-6).to("cuda:0")

    def forward(
        self, hidden_states, past_key_value, attn_bias, attention_mask, is_causal
    ):
        norm_out = self.norm_1(hidden_states)
        attn_output, _, past_key_value = self.attn.forward(
            hidden_states=norm_out,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=None,
            output_attentions=False,
            use_cache=True
        )

        h = hidden_states + attn_output
        out = h + self.ffn.forward(self.norm_2(h))
        return out, None, past_key_value

from transformers.models.mpt.modeling_mpt import MptBlock as OldMptBlock, MptForCausalLM

def fuse_block(model: MptForCausalLM):
    for name, m in model.named_modules():
        if 'mptblock' in m.__class__.__name__.lower():
            m: OldMptBlock

            block = MptBlock(
                model.config.d_model,
                model.config.n_heads,
                m.attn.Wqkv,
                m.attn.out_proj,
                m.ffn
            )

            set_module_name(model, name, block)
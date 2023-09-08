import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from typing import Optional
from awq.quantize.qmodule import WQLinear
import awq_inference_engine
import math
import torch
import torch.nn as nn
import awq_inference_engine
from torch.nn import functional as F

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


class QuantLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)

        self.register_buffer("cos_sin_cache", cache.half(), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ):
        # Apply rotary embedding to the query and key before passing them
        # to the attention op.
        # print(positions.shape, query.shape, key.shape, self.cos_sin_cache.shape)
        query = query.contiguous()
        key = key.contiguous()
        awq_inference_engine.rotary_embedding_neox(
            positions,
            query,
            key,
            self.dim,
            self.cos_sin_cache
        )
        return query, key

class QuantLlamaAttentionFused(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_layer, o_proj, dev, max_position_embeddings):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = self.hidden_size // num_heads
        self.qkv_proj = qkv_layer
        self.o_proj = o_proj
        self.start_pos = 0

        # following fastertransformer definition
        self.cache_v = (
            torch.zeros(
                ( 1, self.n_local_heads, max_position_embeddings, self.head_dim, )
            ).to(dev).half()
        )

        # 8: pack 8 fp16 in FT, if fp32 then use 4
        self.cache_k = (
            torch.zeros(
                ( 1, self.n_local_heads, self.head_dim // 8, max_position_embeddings, 8, )
            ).to(dev).half()
        )
        self.freqs_cis = precompute_freqs_cis(
            hidden_size // num_heads,
            max_position_embeddings * 2,
        ).to(dev)

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

            keys = xk
            values = xv
            past_key_value = (xk, xv) if use_cache else None

            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                scores = scores + attention_mask  # (bs, n_local_heads, slen, cache_len + slen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        else:
            xq = xq[:, 0, :, :]
            xk = xk[:, 0, :, :]
            xv = xv[:, 0, :, :]
            past_key_value = (xk, xv) if use_cache else None
            
            output = awq_inference_engine.single_query_attention(
                xq,
                xk,
                xv,
                self.cache_k,
                self.cache_v,
                None,
                None,
                self.start_pos,
                self.head_dim,
                10000,
                True,
            )
            output = output.reshape(bsz, 1, -1)

        attn_output = self.o_proj(output)

        if use_cache:
            self.start_pos += seqlen
        else:
            self.start_pos = 0

        return attn_output, None, past_key_value

def make_quant_attn(model, dev):
    """
    Replace all LlamaAttention modules with QuantLlamaAttention modules, fusing the q, k, v projections.
    """
    model = model.cpu()
    for name, m in model.named_modules():
        if not isinstance(m, LlamaAttention):
            continue

        q_proj = m.q_proj
        k_proj = m.k_proj
        v_proj = m.v_proj

        qweights = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=0)
        qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=0)
        scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=0)
        # g_idx = torch.cat([q_proj.g_idx, k_proj.g_idx, v_proj.g_idx], dim=0)
        g_idx = None
        bias = (
            torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0)
            if q_proj.bias is not None
            else None
        )

        qkv_layer = WQLinear(
            q_proj.w_bit,
            q_proj.group_size,
            q_proj.in_features,
            q_proj.out_features + k_proj.out_features + v_proj.out_features,
            q_proj.bias is not None,
            q_proj.qweight.device,
        )
        qkv_layer.qweight = qweights
        qkv_layer.qzeros = qzeros
        qkv_layer.scales = scales

        qkv_layer.bias = bias
        qkv_layer.split_k_iters = q_proj.split_k_iters
        # We're dropping the rotary embedding layer m.rotary_emb here. We don't need it in the triton branch.

        if isinstance(m, LlamaAttention):
            attn = QuantLlamaAttentionFused(
                m.hidden_size,
                m.config.num_attention_heads,
                qkv_layer,
                m.o_proj,
                dev,
                m.max_position_embeddings,
            )
        if "." in name:
            parent_name = name.rsplit(".", 1)[0]
            child_name = name[len(parent_name) + 1 :]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ""
            parent = model
            child_name = name

        # print(f"Replacing {name} with quant_attn; parent: {parent_name}, child's name: {child_name}")
        setattr(parent, child_name, attn)
    model = model.to(dev)

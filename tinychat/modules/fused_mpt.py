import math
import torch
from torch import nn
from torch.nn import functional as F
import awq_inference_engine
from tinychat.modules.attn import QuantAttentionFused
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.mpt.modeling_mpt import MptBlock as OldMptBlock, MptForCausalLM

class SharedEmbedding(nn.Embedding):
    def forward(self, input: torch.Tensor, unembed: bool = False) -> torch.Tensor:
        if unembed:
            return F.linear(input, self.weight)
        return super().forward(input)

class MPTBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, qkv_layer, o_proj, mpt_mlp, norm_1, norm_2):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.norm_1 = norm_1
        self.attn = QuantAttentionFused(hidden_size, self.n_heads, qkv_layer, o_proj, dev="cuda:0", max_seq_len=8192, use_alibi=True).to("cuda:0")
        self.norm_2 = norm_2
        self.ffn = mpt_mlp.to("cuda:0")

    def forward(
        self, hidden_states, past_key_value, attn_bias=None, attention_mask=None, is_causal=None
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

class MPTModel(nn.Module):
    def __init__(self, vocab_size, n_layers, d_model, n_heads, blocks, dev):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        self.wte = SharedEmbedding(vocab_size, d_model).to(dev)

        self.blocks: list[MPTBlock] = torch.nn.ModuleList()

        module: OldMptBlock
        for module in blocks:
            self.blocks.append(
                MPTBlock(
                    d_model,
                    n_heads,
                    module.attn.Wqkv,
                    module.attn.out_proj,
                    module.ffn
                ).to(dev)
            )

        self.norm_f = nn.LayerNorm(d_model, eps=1e-5).to(dev)

    @torch.inference_mode()
    def forward(self, input_ids, attn_bias=None, attention_mask=None, is_causal=None, *args, **kwargs):
        _bsz, seqlen = input_ids.shape
        h = self.wte(input_ids)

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=input_ids.device
            )
            mask = torch.triu(mask, diagonal=self.blocks[0].attn.start_pos + 1).type_as(h)

        for layer in self.blocks:
            h, mask, past_key_value = layer(h, None, attention_mask=mask, is_causal=is_causal)
        h = self.norm_f(h)

        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=past_key_value, hidden_states=(), attentions=())

class MPTForCausalLM(nn.Module):
    def __init__(self, vocab_size, n_layers, d_model, n_heads, blocks, dev):
        super().__init__()
        self.transformer = MPTModel(vocab_size, n_layers, d_model, n_heads, blocks, dev)

        for module in self.modules():
            if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
                module.register_parameter("bias", None)

    @torch.inference_mode()
    def forward(self, input_ids, attn_bias=None, attention_mask=None, is_causal=None, *args, **kwargs):
        h = self.transformer(input_ids, attn_bias=attn_bias, attention_mask=attention_mask, is_causal=is_causal)
        output = self.transformer.wte(h, unembed=True)
        return output.float()

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

def fuse_block(model: MptForCausalLM):
    for name, m in model.named_modules():
        if 'mptblock' in m.__class__.__name__.lower():
            m: OldMptBlock

            block = MPTBlock(
                model.config.d_model,
                model.config.n_heads,
                m.attn.Wqkv,
                m.attn.out_proj,
                m.ffn,
                m.norm_1,
                m.norm_2
            )

            set_module_name(model, name, block)

def fuse_transformer(model: MptForCausalLM):
    model.transformer = MPTModel(
        model.config.vocab_size,
        model.config.n_layers,
        model.config.d_model,
        model.config.n_heads,
        model.transformer.blocks,
        next(iter(model.state_dict().values())).device
    )
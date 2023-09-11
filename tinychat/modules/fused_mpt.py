import torch
from torch import nn
from tinychat.modules.attn import QuantAttentionFused
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.mpt.modeling_mpt import MptBlock as OldMptBlock, MptForCausalLM

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
    def __init__(self, vocab_size, blocks, wte, norm_f):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = wte
        self.blocks: list[MPTBlock] = torch.nn.ModuleList(blocks)
        self.norm_f = norm_f

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
            h, _, past_key_value = layer(h, None, attention_mask=mask, is_causal=is_causal)
        h = self.norm_f(h)

        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=past_key_value, hidden_states=(), attentions=())

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
    module: OldMptBlock
    blocks = []

    for module in model.transformer.blocks:
        blocks.append(MPTBlock(
            model.config.d_model,
            model.config.n_heads,
            module.attn.Wqkv,
            module.attn.out_proj,
            module.ffn,
            module.norm_1,
            module.norm_2
        ))

    model.transformer = MPTModel(
        model.config.vocab_size,
        blocks,
        model.transformer.wte,
        model.transformer.norm_f,
    )
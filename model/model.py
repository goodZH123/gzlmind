from transformers import PretrainedConfig


class gzlMindConfig(PretrainedConfig):
    model_type = "glzmind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple, List, Union
from transformers.activations import ACT2FN

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight
    
def precompute_freqs(
        dim: int, 
        end: int = int(32*1024), 
        rope_base: float=1e6,
        rope_scaling: Optional[dict] = None  
        ):
    # 初始化频率
    freqs = 1.0 / (rope_base ** torch.arange(0, dim, 2)[:dim//2].float() / dim)
    attn_factor = 1.0
    if rope_scaling is not None:
        orig_max, factor, attn_factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("attention_factor", 1.0),
            rope_scaling.get("beta_fast", 32),
            rope_scaling.get("beta_slow", 1),
        )

        if end > orig_max:
            inv_dim = lambda b : (dim * math.log(orig_max/(2 * math.pi * b))) / (2 * math.log(rope_base)) 

            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim//2 - 1)

            ramp = torch.clamp((torch.arange(dim//2) - low)/(high - low), 0, 1)

            freqs = freqs * (1 - ramp + ramp * factor)

        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs) * attn_factor
        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor   
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
        return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        x1 = x[..., x.shape[-1]//2:]
        x2 = x[..., :x.shape[-1]//2]
        return torch.cat((-x1, x2), dim=-1)

    q_embed = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    k_embed = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)

    return q_embed, k_embed

def repeqt_kv(x:torch.Tensor, num_repeats: int):
    if num_repeats == 1:
        return x
    batch_size, seq_len, heads, head_dim = x.shape
    x = x[:,:,:,None,:].expand(batch_size, seq_len, heads, num_repeats, head_dim).reshape(batch_size, seq_len, heads * num_repeats, head_dim)
    return x

class Attention(nn.Module):
    def __init__(self, args: gzlMindConfig):
        super().__init__()
        self.num_key_value_heads = (args.num_attention_heads 
                                    if args.num_key_value_heads is None 
                                    else args.num_key_value_heads)
        
        assert args.num_attention_heads % self.num_key_value_heads == 0 

        self.local_heads = args.num_attention_heads
        self.local_kv_heads = self.num_key_value_heads
        self.n_rep = self.local_heads // self.local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, self.local_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.local_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.local_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.local_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)    
        self.res_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = args.flash_attention and hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                attention_mask: Optional[torch.Tensor] = None, 
                position_embeddings: Optional[torch.Tensor] = None,
                use_cache: bool = False):
        # 计算query,key,value对query、key加上位置编码，并且将key和value进行重复以适应多头注意力机制
        # query与key相乘得到注意力权重，并通过掩码进行调整，经过softmax得到最终的注意力分布，最后将注意力分布与value相乘得到输出，并通过线性变换得到最终的结果

        batch_size, seq_len, _ = x.shape
        
        qx = self.q_proj(x).view(batch_size, seq_len, self.local_heads, self.head_dim)
        kx = self.k_proj(x).view(batch_size, seq_len, self.local_kv_heads, self.head_dim)
        vx = self.v_proj(x).view(batch_size, seq_len, self.local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        qx, kx = apply_rotary_pos_emb(qx, kx, cos, sin)

        if past_kv is not None:
            kx = torch.cat([past_kv[0], kx], dim=1)
            vx = torch.cat([past_kv[1], vx], dim=1)
        past_kv = (kx, vx) if use_cache else None

        qx = qx.transpose(1, 2)
        kx = repeqt_kv(kx, self.n_rep).transpose(1, 2)
        vx = repeqt_kv(vx, self.n_rep).transpose(1, 2)

        if self.flash and (seq_len > 1) and (past_kv is None) and ((attention_mask is None) or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(qx, kx, vx, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            attn_weights = torch.matmul(qx, kx.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights[:,:,:,-seq_len:] += torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=attn_weights.device), diagonal=1)
            if attention_mask is not None:
                attention_mask = ((1 - attention_mask) * -1e9).unsqueeze(1).unsqueeze(2)
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1).type_as(qx)
            attn_weights = self.attn_dropout(attn_weights)
            output = torch.matmul(attn_weights, vx)
            output = output.transpose(1,2).reshape(batch_size, seq_len, self.local_heads * self.head_dim)
        output = self.o_proj(output)
        output = self.res_dropout(output)
        return output, past_kv
    
class FeedForward(nn.Module):
    def __init__(self, args: gzlMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64 * ((intermediate_size + 64 -1) // 64)

        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]
    
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        down = self.down_proj(self.act_fn(gate) * up)
        output = self.dropout(down)
        return output

class MOEfeedForward(nn.Module):
    def __init__(self, config: gzlMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_routed_experts)])
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.reshape(-1, hidden_size)
        scores = self.gate(x_flat)
        scores = F.softmax(scores, dim=-1)
        topk_weight, topk_indices = torch.topk(scores, self.config.num_experts_per_tok, dim=-1)
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-9)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                expert_input = x_flat[token_idx]
                y.index_add_(0, token_idx, (expert(expert_input) * topk_weight[mask]).to(y.dtype))        
            elif self.training:
                y[0,0] += 0.0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.aux_loss_alpha > 0:
            load = F.one_hot(topk_indices, self.config.n_routed_experts).float().mean(dim=0)
            self.aux_loss = (load * scores.mean(dim=0)).sum() * self.config.n_routed_experts * self.config.aux_loss_alpha
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.reshape(batch_size, seq_len, hidden_size)

# class gzlMindBlock(nn.Module):
#     def __init__(self, config: gzlMindConfig):
#         super().__init__() 
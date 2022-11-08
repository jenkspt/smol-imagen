from typing import Callable, Optional
from functools import partial
from math import prod, log
import jax
import jax.numpy as jnp
import flax.linen as nn


class SiLU(nn.Module):

    def __call__(self, x):
        return nn.silu(x)


def positional_encoding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = jnp.exp(-log(max_period) * jnp.arange(half) / half)
    args = timesteps[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


class PositionalEncoding(nn.Module):
    dim: int
    max_period: int = 10000

    @nn.compact
    def __call__(self, x):
        return positional_encoding(x, self.dim, self.max_period) 


@jax.checkpoint
def attention_fn(q, k, v, mask):
    return nn.attention.dot_product_attention(q, k, v, mask=mask, deterministic=True)


class AttentionBlock(nn.Module):

    num_head_channels: int
    #attention_fn: Callable = nn.attention.dot_product_attention
    attention_fn: Callable = attention_fn

    @nn.compact
    def __call__(self, x,
            cond_sequence=None,          # [B, kv_t, D]
            padding=None,                # [B, kv_t]
        ):
        b, *spatial, c = x.shape
        nd = len(spatial)
        t = prod(spatial)

        assert c % self.num_head_channels == 0
        num_heads = c // self.num_head_channels

        mask = None
        # normalize image features and project to q, k, v
        h = nn.GroupNorm()(x)
        h = nn.Conv(c*3, (1,)*nd, padding=0, name='qkv')(h)
        qkv = h.reshape(b, t, num_heads, c*3 // num_heads)
        q, k, v = jnp.array_split(qkv, 3, -1)

        if cond_sequence is not None:
            # normalize conditioning sequence, project and concatenate to kv
            ctx_kv = nn.LayerNorm()(cond_sequence)  # --> [b, kv_t, C]
            ctx_kv = nn.Conv(c*2, (1,), padding=0, name='context_kv')(ctx_kv)
            ctx_kv = ctx_kv.reshape(b, -1, num_heads, c*2 // num_heads)
            ctx_k, ctx_v = jnp.array_split(ctx_kv, 2, -1)
            k = jnp.concatenate([k, ctx_k], -3)
            v = jnp.concatenate([v, ctx_v], -3)

            if padding is not None: 
                kv_t = padding.shape[-1]
                # image features are always used so mask t x t matrix of 1's
                mask = jnp.ones((b, 1, t, t), dtype=bool)
                padding = jnp.broadcast_to(padding[:, None, None, :], (b, 1, t, kv_t))
                mask = jnp.concatenate([mask, padding.astype(bool)], -1)    # --> [b, 1, q_t, q_t + kv_t]

        # attention weights are masked out if their mask value is `False`
        h = self.attention_fn(q, k, v, mask=mask)
        h = h.reshape(x.shape)
        h = nn.Conv(c, (1,) * nd, padding=0, name='project_out', kernel_init=nn.initializers.zeros)(h)
        return h + x
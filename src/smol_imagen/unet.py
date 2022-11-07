from typing import Tuple, Optional
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp

from .common import AttentionBlock, SiLU


class Upsample(nn.Module):
    """
    :param channels: if specified, adds a Conv layer with channels after
        the resize upsampling
    """
    channels: Optional[int] = None

    @nn.compact
    def __call__(self, x):

        b, *spatial, c = x.shape

        x = jax.image.resize(
            x,
            shape=(b, *(s * 2 for s in spatial), c),
            method="nearest",
            antialias=False,
        )

        if self.channels:
            x = nn.Conv(self.channels, (3,) * len(spatial))(x)
        return x


class Downsample(nn.Module):

    """
    Downsample input by factor of 2 in spatial dimensions
    :param channels: if specified, adds a Conv layer with channels before
        the downsample average pooling
    """

    channels: Optional[int] = None

    @nn.compact
    def __call__(self, x):
        nd = x.ndim - 2
        if self.channels:
            return nn.Conv(self.channels, (3,) * nd, strides=2)(x)
        else:
            return nn.avg_pool(x, (2,) * nd, (2,) * nd)


class ResBlock(nn.Module):

    """
    A residual block that can optionally change the number of channels.
    :param channels: if specified, the number of out channels.
    :param dropout: the rate of dropout.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    channels: Optional[int] = None
    dropout: float = 0.0
    deterministic: Optional[bool] = None
    use_conv: bool = False
    use_scale_shift_norm: bool = False
    up: bool = False
    down: bool = False

    @nn.compact
    def __call__(self, x, cond=None, deterministic=None):

        assert not (self.up and self.down), "Must choose either up or down"
        nd = x.ndim - 2  # 1D, 2D or 3D
        channels = self.channels or x.shape[-1]
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        h = nn.GroupNorm()(x)
        h = jax.nn.silu(h)
        if self.up:
            h = Upsample()(h)
            x = Upsample()(x)
        elif self.down:
            h = Downsample()(h)
            x = Downsample()(x)

        h = nn.Conv(channels, (3,) * nd)(h)

        h = nn.GroupNorm()(h)

        if cond is not None:

            emb_out = nn.Sequential(
                [
                    nn.LayerNorm(),
                    SiLU(),
                    nn.Dense(2 * channels if self.use_scale_shift_norm else channels),
                ],
                name="emb_layers",
            )(cond)

            emb_out = emb_out.reshape(x.shape[0], *((1,) * nd), -1)

            if self.use_scale_shift_norm:
                scale, shift = jnp.array_split(emb_out, 2, -1)
                h = h * (1 + scale) + shift
            else:
                h = h + emb_out

        h = jax.nn.silu(h)
        h = nn.Dropout(self.dropout)(h, deterministic=deterministic)
        h = nn.Conv(channels, (3,) * nd, kernel_init=nn.initializers.zeros)(h)

        if self.use_conv:
            x = nn.Conv(channels, (3,) * nd, name="skip_connection")(x)
        elif h.shape[-1] != x.shape[-1]:
            x = nn.Conv(channels, (1,) * nd, name="skip_connection")(x)

        return h + x


class UNetModel(nn.Module):

    model_channels: int
    out_channels: int
    num_res_blocks: int
    attention_resolutions: Tuple[int, ...]
    dropout: float = 0.0
    deterministic: Optional[bool] = None
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    conv_resample: bool = True
    num_head_channels: int = 64
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False

    @nn.compact
    def __call__(self, x, cond=None, cond_sequence=None, padding=None):
        nd = x.ndim - 2     # spatial dimensions
        _ResBlock = partial(ResBlock,
            dropout=self.dropout,
            deterministic=self.deterministic or self.dropout == 0.0,
            use_scale_shift_norm=self.use_scale_shift_norm)

        channels = self.channel_mult[0] * self.model_channels

        h = nn.Conv(channels, (3,)*nd, name="input_conv")(x)
        hs, ds = [h], 1
        for level, mult in enumerate(self.channel_mult):
            channels = mult * self.model_channels
            for _ in range(self.num_res_blocks):
                h = _ResBlock(channels)(h, cond)

                if ds in self.attention_resolutions:
                    h = AttentionBlock(self.num_head_channels)(h, cond_sequence, padding)
                hs.append(h)

            if level != len(self.channel_mult) - 1:
                if self.resblock_updown:
                    h = _ResBlock(channels, down=True)(h, cond)
                else:
                    h = Downsample(channels if self.conv_resample else None)(h)
                hs.append(h)
                ds *= 2

        # middle blocks
        h = _ResBlock(channels)(h, cond)
        h = AttentionBlock(self.num_head_channels)(h, cond_sequence, padding)
        h = _ResBlock(channels)(h, cond)

        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            channels = mult * self.model_channels
            for i in range(self.num_res_blocks + 1):
                h = jnp.concatenate([h, hs.pop()], -1)
                h = _ResBlock(channels)(h, cond)
                if ds in self.attention_resolutions:
                    h = AttentionBlock(self.num_head_channels)(h, cond_sequence, padding)

                if level and i == self.num_res_blocks:
                    if self.resblock_updown:
                        h = _ResBlock(channels, up=True)(h, cond)
                    else:
                        h = Upsample(channels if self.conv_resample else None)(h)
                    ds //= 2
        assert not hs
        h = nn.Sequential(
            [
                nn.GroupNorm(),
                SiLU(),
                nn.Conv(self.out_channels, (3,)*nd, kernel_init=nn.initializers.zeros),
            ],
            name="output_layers",
        )(h)
        return h
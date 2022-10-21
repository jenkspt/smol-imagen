import jax
import jax.numpy as jnp

import flax.linen as nn

from smol_imagen.unet import (
    Upsample,
    Downsample,
    ResBlock,
    AttentionBlock,
    UNetModel
)

from smol_imagen.models import ImagenCLIP64


def init_apply(key, m: nn.Module, *args, rngs={}):
    params = m.init({'params': key, **rngs}, *args)
    return m.apply(params, *args, rngs=rngs)


def test_Upsample(getkey):
    
    x = jax.random.normal(getkey(), (3, 8, 8, 16))
    y = init_apply(getkey(), Upsample(), x)
    assert y.shape == (3, 16, 16, 16)
    y = init_apply(getkey(), Upsample(8), x)
    assert y.shape == (3, 16, 16, 8)


def test_Downsample(getkey):
    x = jax.random.normal(getkey(), (3, 16, 16, 8))
    y = init_apply(getkey(), Downsample(), x)
    assert y.shape == (3, 8, 8, 8)
    y = init_apply(getkey(), Downsample(16), x)
    assert y.shape == (3, 8, 8, 16)


def test_ResBlock(getkey):
    x = jax.random.normal(getkey(), (3, 16, 16, 32))
    y = init_apply(getkey(), ResBlock(deterministic=True), x, None)
    assert y.shape == x.shape
    layer = ResBlock(dropout=0.2, deterministic=False)
    y = init_apply(getkey(), layer, x, None, rngs={'dropout': getkey()})
    assert y.shape == x.shape
    layer = ResBlock(channels=64, dropout=0.0, deterministic=True)
    y = init_apply(getkey(), layer, x, None)
    assert y.shape == (3, 16, 16, 64)
    layer = ResBlock(use_conv=True, dropout=0.0, deterministic=True)
    y = init_apply(getkey(), layer, x, None)
    assert y.shape == (3, 16, 16, 32)
    cond = jax.random.normal(getkey(), (3, 64))
    y = init_apply(getkey(), ResBlock(deterministic=True), x, cond)
    assert y.shape == x.shape


def test_AttentionBlock(getkey):
    x = jax.random.normal(getkey(), (3, 8, 8, 32))
    layer = AttentionBlock(2)
    y = init_apply(getkey(), layer, x, None)
    assert y.shape == x.shape
    assert jnp.allclose(y, x)   # because of zero init + residual
    cond_sequence = jax.random.normal(getkey(), (3, 9, 64))
    layer = AttentionBlock(2)
    y = init_apply(getkey(), layer, x, cond_sequence)
    assert y.shape == x.shape
    assert jnp.allclose(y, x)   # because of zero init + residual


def test_UNetModel(getkey):

    model = UNetModel(
        model_channels = 32,
        out_channels = 3,
        num_res_blocks = 1,
        attention_resolutions = (16, 8),
        dropout = 0.0,
        channel_mult = (1, 1, 2),
        conv_resample = True,
        num_heads = 1,
        use_scale_shift_norm = True,
        resblock_updown = False,
    )
    x = jax.random.uniform(getkey(), (2, 32, 32, 3), minval=-1, maxval=1)
    y = init_apply(getkey(), model, x, None, None)
    assert y.shape == x.shape
    assert jnp.allclose(y, jnp.zeros_like(x))

    cond = jax.random.normal(getkey(), (2, 64))
    cond_sequence = jax.random.normal(getkey(), (2, 9, 64))
    y = init_apply(getkey(), model, x, cond, cond_sequence)
    assert y.shape == x.shape
    assert jnp.allclose(y, jnp.zeros_like(x))


def test_ImagenCLIP64(getkey):
    model = ImagenCLIP64(64, num_heads=2)
    params = model.init(getkey())
    padding = (jax.random.uniform(getkey(), (2, 77,)) > .5).astype(jnp.int32).sort()[::-1]
    x = jax.random.uniform(getkey(), (2, 64, 64, 3), minval=-1, maxval=1)
    t = jax.random.uniform(getkey(), (2,))
    cond_sequence = jax.random.normal(getkey(), (2, 77, 768))
    y = model.apply(params, x, t, cond_sequence, padding)
    assert x.shape == y.shape
    assert jnp.allclose(y, jnp.zeros_like(x))
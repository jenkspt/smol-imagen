from typing import Tuple, Optional

import jax.numpy as jnp
import flax.linen as nn
from .unet import UNetModel
from .common import SiLU, PositionalEncoding


class ImagenCLIP64(UNetModel):
    """ CLIP conditioned Imagen 64x64 base model unet """

    # override unet defaults
    out_channels: int = 3
    num_res_blocks: int = 3
    attention_resolutions: Tuple[int, ...] = (2, 4, 8)
    deterministic: Optional[bool] = True
    channel_mult: Tuple[int, ...] = (1, 2, 3, 4)
    num_head_channels: int = 64
    use_scale_shift_norm: bool = True

    @nn.compact
    def __call__(self, x, t, encoded_text, padding):
        """
        :param x:
        :param t:
        :param encoded_text:
        :param padding: 
        """
        assert x.ndim == 4
        assert t.ndim == 1
        assert encoded_text.ndim == 3
        assert padding.ndim == 2
        assert x.shape[0] == t.shape[0] == encoded_text.shape[0] == padding.shape[0]
        
        # In the CLIP conditioned version, CLIP already has a pooled output of cond_sequence,
        # so we use that
        # Clip uses the end token as the pooled embedding output
        # so to get the pooled value we either take the last text encoding or use the padding to
        # find the last non-padded embedding
        B = encoded_text.shape[0]
        last_index = padding.argmin(-1) - 1
        pooled_text = encoded_text[jnp.arange(B), last_index, :]

        time_embed_dim = 4 * self.model_channels
        time_embed = nn.Sequential([
            PositionalEncoding(self.model_channels),
            nn.Dense(time_embed_dim),
            SiLU(),
            nn.Dense(time_embed_dim),
        ])(t)
        cond = time_embed + nn.Dense(time_embed_dim)(pooled_text)

        return super().__call__(x, cond, encoded_text, padding)


    def init(self, key, image_size: int=64, cond_dim: int=768, seq_length: int=77):
        z_t = jnp.zeros((1, image_size, image_size, 3))
        t = jnp.zeros((1,))
        encoded_text = jnp.zeros((1, seq_length, cond_dim))
        padding = jnp.zeros((1, seq_length))

        params = super().init(key, z_t, t, encoded_text, padding)
        return params


class ImagenCLIP256:

    def __init__(self):
        raise NotImplementedError()



class ImagenCLIP1024:

    def __init__(self):
        raise NotImplementedError
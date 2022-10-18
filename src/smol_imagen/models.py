from typing import Tuple, Optional

import flax.linen as nn
from .unet import UNetModel
from .common import SiLU, PositionalEncoding


class ImagenCLIP64(UNetModel):
    """ CLIP conditioned Imagen 64x64 base model unet """

    # override unet defaults
    out_channels: int = 3
    num_res_blocks: int = 3
    attention_resolutions: Tuple[int, ...] = (32, 16, 8)
    deterministic: Optional[bool] = True
    channel_mult: Tuple[int, ...] = (1, 2, 3, 4)
    num_heads: int = 8
    use_scale_shift_norm: bool = True

    @nn.compact
    def __call__(self, x, t, cond, cond_sequence, padding=None):
        # In the CLIP conditioned version, CLIP already has a pooled output of cond_sequence,
        # so we use that
        time_embed_dim = 4 * self.model_channels
        time_embed = nn.Sequential([
            PositionalEncoding(self.model_channels),
            nn.Dense(time_embed_dim),
            SiLU(),
            nn.Dense(time_embed_dim),
        ])(t)
        cond = time_embed + nn.Dense(time_embed_dim)(cond)

        return super().__call__(x, cond, cond_sequence, padding)



class ImagenCLIP256:

    def __init__(self):
        raise NotImplementedError()



class ImagenCLIP1024:

    def __init__(self):
        raise NotImplementedError
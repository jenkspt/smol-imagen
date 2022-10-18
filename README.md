Smol Imagen
===========
Unofficial implementation of Google's Imagen: ["Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding"](https://arxiv.org/abs/2205.11487)

> Imagen, a text-to-image diffusion model with an unprecedented degree of photorealism and a deep level of language understanding

## Setup

```sh
pip install git+https://github.com/jenkspt/smol-imagen.git
```

## Usage

```python
from smol_imagen import ImagenCLIP64

model = ImagenCLIP64()

```


```bibtex
@misc{https://doi.org/10.48550/arxiv.2205.11487,
  doi = {10.48550/ARXIV.2205.11487},
  
  url = {https://arxiv.org/abs/2205.11487},
  
  author = {Saharia, Chitwan and Chan, William and Saxena, Saurabh and Li, Lala and Whang, Jay and Denton, Emily and Ghasemipour, Seyed Kamyar Seyed and Ayan, Burcu Karagol and Mahdavi, S. Sara and Lopes, Rapha Gontijo and Salimans, Tim and Ho, Jonathan and Fleet, David J and Norouzi, Mohammad},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
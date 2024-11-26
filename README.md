# Unsupervised Video Denoisers used in the paper _"Atomic Resolution Observations of Nanoparticle Surface Dynamics and Instabilities Enabled by Deep Denoising"_

Check out the [preprint](https://arxiv.org/abs/2407.17669) of the paper!

## Introduction

Understanding the structural dynamics of nanoparticle surfaces at the atomic scale is crucial for advancing applications in areas like diffusion, reactivity, and catalysis. However, studying these dynamics poses a major challenge, as it demands both high spatial and temporal resolution. While ultrafast transmission electron microscopy (TEM) can provide picosecond temporal resolution, its spatial resolution is limited to the nanometer scale. On the other hand, conventional TEM, when combined with high-readout-rate electron detectors, is capable of achieving atomic spatial resolution with millisecond-level time resolution. Unfortunately, capturing atomic structure with millisecond resolution requires lowering electron dose rates to prevent beam-induced damage, leading to severe noise in the images and loss of structural clarity.

This repository provides an unsupervised denoising framework based on artificial intelligence developed to overcome these limitations. Our denoising pipeline enables atomic-resolution visualization of metal nanoparticle surfaces by leveraging Deep Learning models to exploit the correllation among neighboring pixels and/or frames. 

The provided pipeline can be easilly adapted in order to denoise other data types, either images or videos, beyond TEM videos in `.tif` or `.npy` file.

## Denoisers:
The present package incorporates the following denoisers:
* **UDVD (Unsupervised Deep Video Denoising)** from [Paper](https://arxiv.org/abs/2011.15045), [GitHub](https://github.com/sreyas-mohan/udvd)

## Usage
### Installation
```shell
git clone https://github.com/adriamm98/blind_video_denoisers
cd blind_video_denoisers
pip install -r requirements.txt

```

### Command
```shell
python denoise.py\
     --data path_to_tiff_file 
     --model UDVD 
     --num-epochs 50
     --batch-size 1
     --image-size 256
```
### Arguments
* `data` (required): Full path to the `.tif` or `.npy` file containing the video to be denoised. The `.npy` file should have shape of (frames, x, y).
* `model`: Model name. Options are `UDVD`, `N2N`, `N2S`, or `UDVD_sf`. Default is `UDVD`.
* `num-epochs` Number of training epochs(default: 50).
* `batch-size`: Number of images per batch for training (default: 1). Adjust based on available memory.
* `image-size`: Size of the square image patches used for training (default: 256). For N2N, a larger size, such as 512, is recommended to compensate the downsampling step.
* `save-format`: Select the format of denoised file (`tif` or `npy`). Default is `tif`.

### Example

The provided `PtCeO2_6.tif` video can be denoised by running the following commands:

```shell
python denoise.py --data ./examples/PtCeO2_6.tif # denoise with the UDVD model
```

### Citation

If you use this code, please cite our work:

```bibtex
@inproceedings{crozier2024atomic,
  title={Atomic Resolution Observations of Nanoparticle Surface Dynamics and Instabilities Enabled by Artificial Intelligence},
  author={Crozier, Peter A. and Leibovich, Matan and Haluai, Piyush and Tan, Mai and Thomas, Andrew M. and Vincent, Joshua and Mohan, Sreyas and Marcos Morales, Adria and Kulkarni, Shreyas A. and Matteson, David S. and Wang, Yifan and Fernandez-Granda, Carlos},
  year={2024},
  note={arXiv preprint arXiv:2407.17669},
  url={https://arxiv.org/abs/2407.17669}
}

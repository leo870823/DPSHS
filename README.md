# [WACV 2024] Deep Plug-and-play Nighttime Non-blind Deblurring with Saturated Pixel Handling Schemes

## Abstract
Due to the setting of shutter speeds, over-exposed blurry images can often be seen in nighttime photography. Although image deblurring is a classic problem in image restoration, state-of-the-art methods often fail in nighttime cases with saturated pixels. The primary reason is that those pixels are out of the sensor range and thus violate the assumption of the linear blur model. To address this issue, we propose a new nighttime non-blind deblurring algorithm with saturated pixel handling schemes, including a pixel stretching mask, an image segment mask, and a saturation awareness mechanism (SAM). Our algorithm achieves superior results by strategically adjusting mask configurations, making our method robust to various saturation levels. We formulate our task into two new optimization problems and introduce a unified framework based on the plug-and-play alternating direction method of multipliers (PnP-ADMM). We also evaluate our approach qualitatively and quantitatively to demonstrate its effectiveness. The results show that the proposed algorithm recovers sharp latent images with finer details and fewer artifacts than other state-of-the-art deblurring methods.

## Dependencies
* Python 3.8.12 (tested with anaconda3)
* PyTorch 1.10.0 (CUDA=11.3)

## Install Packages
pip3 install -r requirement.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html

## Download the DRUNet pre-trained model from [DPIR](https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D)
Please put the drunet_color.pth below model/pre-trained.

## Benchmark Datasets
* [The Saturated Dataset from Hu et al.](https://eng.ucmerced.edu/people/zhu/CVPR14_lightstreak.html)
* [Low-illumination Dataset from Pan et al.](https://pan.baidu.com/s/1O2AezDHc64GzHyU_U7BX4g)
* [Night Dataset from Chen et al.](https://drive.google.com/file/d/1C7J9rn2xbeJ4-Aom4KEQJdpFyBd2M4Zv/view)
* Download all datasets and then put them below the Benchmark/ folder.

## Reproduce the Saturated Dataset
```
python3 test_deblur_benchmark.py -dataset Hu -log Log/Compare_Saturated/
```

## Reproduce the Low-illumination Dataset
```
python3 test_deblur_benchmark.py -dataset Pan -log Log/Compare_Low/
```

## Reproduce the Night Dataset
```
python3 test_deblur_benchmark.py -dataset Chen -log Log/Compare_Night/
```

## Reproduce the Real World Blurry Images
```
python3 test_deblur_real_world.py
```

## Note
* This implementation incorporates code from [Pytorch-template](https://github.com/victoresque/pytorch-template) and [DPIR](https://github.com/cszn/DPIR), rspectively.

## Citation
```bibtex
@inproceedings{shu2024deep,
  title={Deep Plug-and-play Nighttime Non-blind Deblurring with Saturated Pixel Handling Schemes},
  author={Shu, Hung-Yu and Lin, Yi-Hsien and Lu, Yi-Chang},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1538--1546},
  year={2024}
}

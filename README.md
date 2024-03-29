# DPSHS
This is a pytorch implementation of our research. Please refer to our paper for more details:
Deep Plug-and-play Nighttime Non-blind Deblurring with Saturated Pixel Handling Schemes


## Dependencies
* python 3.8.12 (tested with anaconda3)
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
* This implementation is based on the implementation of [Pytorch-template](https://github.com/victoresque/pytorch-template) and [DPIR](https://github.com/cszn/DPIR).
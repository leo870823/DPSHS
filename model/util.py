import cv2
import piq
import torch
import numpy as np
import math
import pandas as pd
from os import mkdir,makedirs

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        # ignore trivial row
        # self._data = self._data.drop(index=(""))
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, epoch=0,n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value, epoch)
        
        # handle non-exist key
        if key not in self._data.total:
            print("missing key {}".format(key))
            df = pd.DataFrame([[0,0,0]],
                               columns=['total', 'counts', 'average'],
                               index=[key])
            self._data = pd.concat([self._data,df])

        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]


    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def matlabPSNR(tar_img, prd_img,peak_val = 255.0,err_map = False):
	imdff = prd_img.astype(np.float64) - tar_img.astype(np.float64)
	rmse = np.mean(imdff**2)
	if rmse !=0:
		ps = 10 * np.log10(peak_val**2 / rmse)
	else:
		ps = math.inf

	if err_map:
		if len(imdff.shape)  == 3:
			return (ps,np.expand_dims(np.sum(np.absolute(imdff),axis=2),axis=-1))
		else:
			return (ps,np.expand_dims(imdff,axis=-1))
	else:
		return ps

def torchPSNR(tar_img, prd_img):
	imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
	rmse = (imdff**2).mean().sqrt()
	ps = 20 * torch.log10(1 / rmse)
	return ps


def tensor_to_np_multiple(img):
	#img = img.mul(1.0).byte()
	img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img 
 

def save_img_with_PSNR(filepath, img, golden, kernel_np, PSNR = 100 ,SSIM = 1.0 ,PSNR_flag=True, WEIGHT_DICT={}, mode=""):
	if PSNR_flag:
		#PSNR = matlabPSNR(img, golden)
		#SSIM = skimage_ssim(img, golden,multichannel=True,data_range=255.0)
		PSNR_TAG = "%s PSNR: %.4f SSIM: %.4f" % (mode, PSNR,SSIM)
		for key in WEIGHT_DICT:
			PSNR_TAG = PSNR_TAG + " {} : {}".format(key, WEIGHT_DICT[key])
	else:
		cv2.imwrite(filepath, img)
		return
	if kernel_np is not None:
		kernel_np = kernel_np * 255.0 / np.max(kernel_np)
		img = cv2.putText(img, PSNR_TAG, (kernel_np.shape[0], kernel_np.shape[1]),
						  cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)
		# Handle kernel flag is None case
		if img.shape[-1] == 3:
			img[0:kernel_np.shape[0], 0:kernel_np.shape[1], :] = kernel_np
		else:
			img[0:kernel_np.shape[0], 0:kernel_np.shape[1]] = kernel_np[:, :, 0:1]
	else:
		img = cv2.putText(img, PSNR_TAG, (35, 35), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)
	cv2.imwrite(filepath, img)
	if PSNR_flag:
		return PSNR,SSIM


def mkdirs(paths):
    if isinstance(paths, str):
        makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            makedirs(path, exist_ok=True)
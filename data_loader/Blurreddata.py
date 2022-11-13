import cv2
import os
import glob
import numpy as np
import math
import random
import torch
import torch.utils.data as data
import torch.nn.functional as F
from scipy.io import loadmat
from .blur_util import *
from pathlib import Path


#
# Handle our own blurred process
#
class Dataset(data.Dataset):
	def __init__(self, sharp,kernel,noise_level):
		#self.blurred=blurred
		self.sharp  =sharp
		self.kernel =kernel
		self.noise_level  =  noise_level #(noise == "True")
		self.file_list = sorted([name for name in os.listdir(self.sharp) if os.path.isfile(os.path.join(self.sharp, name))])
		self.kernel_list = sorted([name for name in os.listdir(self.kernel) if os.path.isfile(os.path.join(self.kernel, name))])
		self.sharp_num=len(self.file_list)
		self.kernel_num=len(self.kernel_list)
		
		#self.blurred_num=len([name for name in os.listdir(self.blurred) if os.path.isfile(os.path.join(self.blurred, name))])
		# Dataset flag
		self.BLURRED_FLAG=True
		self.KERNEL_FLIP_FLAG=False
		self.summary()
		
	def summary(self):
		print("Read training data: %d kernel pattern,%d sharp image"%(self.kernel_num,self.sharp_num))
		#print("Blurred image from:",blurred)
		print("Blurred image from synthetic image")
		print("Kernel  image from:",self.kernel)
		print("Sharp   image from:",self.sharp)
		if self.noise_level:
			print("Adding noise in blurred image")
		#assert(self.blurred_num==self.sharp_num*self.kernel_num)
	def __len__(self) -> int:
		return self.sharp_num*self.kernel_num



	def RGBConv2D(self,image,kernel):
		RGB_image=torch.zeros(image.shape).to(self.device)
		# pad last dim by (1, 1) and 2nd to last by (2, 2)
		p1d=(int(kernel.shape[3]/2),int(kernel.shape[3]/2),int(kernel.shape[2]/2),int(kernel.shape[2]/2))
		image= F.pad(image, p1d, "circular")
		#image= F.pad(image, p1d, "reflect")
		for i in range(0,image.shape[1]):
			RGB_image[:,i:i+1,:,:]= F.conv2d(image[:,i:i+1,:,:], kernel)
		return RGB_image

	def set_data_path(self,index):
		sharp  =os.path.join(self.sharp  ,'%d.jpg'%(int(index/self.kernel_num)))
		kernel =os.path.join(self.kernel ,'linear_%d.jpg'%(int(index%self.kernel_num)))#'{:06d}.jpg'
		return sharp,kernel

	def __getitem__(self, index: int) :
		#handle path
		#blurred=os.path.join(self.blurred,'%d.jpg'%(index))
		sharp,kernel=self.set_data_path(index)
		#kernel =os.path.join(self.kernel ,'%d.png'%(int(index%self.kernel_num)))#'{:06d}.jpg'
		#Read image
		#img_blurred=cv2.imread(blurred)
		img_sharp  = cv2.imread(sharp)/255.0
		#img_sharp  = np.power(img_sharp,2.2)
		#resize
		#dim=(int(img_sharp.shape[0]/10)*10,int(img_sharp.shape[1]/10)*10)
		#img_sharp=img_sharp[0:dim[0],0:dim[1],:]
		#img_sharp = cv2.resize(img_sharp, dim, interpolation = cv2.INTER_AREA)
		img_kernel =cv2.imread(kernel,cv2.IMREAD_GRAYSCALE)
		img_kernel =img_kernel/np.sum(img_kernel)

		if self.KERNEL_FLIP_FLAG:
			img_kernel=np.flip(img_kernel, [0,1])

		if self.BLURRED_FLAG:
			if img_sharp.shape[-1] == 3:
				img_blurred=color_convolution(img_sharp,img_kernel) # synthetic blurred image
			else:
				img_blurred=convolution(img_sharp,img_kernel) # synthetic blurred image
		else:
		   img_blurred= cv2.imread(blurred)/255.0
		#Rotate image
		if img_blurred.shape[0]>img_blurred.shape[1]:
		   img_blurred=cv2.rotate(img_blurred, cv2.ROTATE_90_CLOCKWISE) 
		   img_sharp=cv2.rotate(img_sharp, cv2.ROTATE_90_CLOCKWISE) 
		   img_kernel=cv2.rotate(img_kernel, cv2.ROTATE_90_CLOCKWISE)
		#Dimension swap
		img_blurred=np.transpose(img_blurred, (2,0,1))
		img_sharp  =np.transpose(img_sharp, (2,0,1))
		#Add noise 
		if self.noise_level :
			#img_blurred=img_blurred+np.random.normal(loc=0.0, scale=1.0/255.0,size=img_blurred.shape)
			#print("add {} % noise level".format(self.noise_level))
			img_blurred=(img_blurred+np.random.normal(loc=0.0, scale=self.noise_level/255.0,size=img_blurred.shape))#/255.0
		else:
			img_blurred=img_blurred#/255.0
		#Dimension extension
		img_kernel=torch.FloatTensor(img_kernel.copy()).repeat(3, 1, 1)
		#print("sharp  ",np.max(img_sharp))
		#print("blurred",np.max(img_blurred))
		# gamma correction
		#img_sharp  = np.power(img_sharp,1/2.2)
		#img_blurred  = np.power(img_blurred,1/2.2)
		return img_kernel,torch.FloatTensor(img_sharp),torch.FloatTensor(img_blurred)


#####################
# Real World Dataset
#####################
class Real_World_Dataset(Dataset):
	def __init__(self, blurred, kernel, noise_level):
		super().__init__(blurred, kernel, noise_level)
		self.blurred, self.kernel = blurred, kernel
		self.KERNEL_FLIP_FLAG = False
		self.file_list = sorted([name for name in os.listdir(self.blurred) if os.path.isfile(os.path.join(self.blurred, name))])
		self.kernel_list = sorted([name for name in os.listdir(self.kernel) if os.path.isfile(os.path.join(self.kernel, name))])
		self.blurred_num=len(self.file_list)
		self.kernel_num=len(self.kernel_list)


	
	def __len__(self):
		return len(self.file_list)

	def set_data_path(self,index):
		blurred  = os.path.join(self.blurred  ,self.file_list  [index])
		kernel = os.path.join(self.kernel ,self.kernel_list [index] )
		return blurred,kernel

	def __getitem__(self, index: int) :
		#handle path
		blurred,kernel=self.set_data_path(index)
		img_blurred  = cv2.imread(blurred)/255.0
		img_kernel =cv2.imread(kernel,cv2.IMREAD_GRAYSCALE)
		kh,kw = img_kernel.shape
		if kh != kw:
			k_large = max(kh,kw)
			print("Pad kerenl to {}".format(k_large))
			kernel_pad = np.zeros((k_large,k_large))
			half_pad = k_large//2
			h = half_pad-kh//2
			w = half_pad-kw//2 
			kernel_pad[h:h+kh,w:w+kw] = img_kernel
			img_kernel = kernel_pad

		img_kernel =img_kernel/np.sum(img_kernel)

		if self.KERNEL_FLIP_FLAG:
			img_kernel=np.flip(img_kernel, [0,1])

		#Dimension swap
		img_blurred  =np.transpose(img_blurred, (2,0,1))
		img_blurred = torch.FloatTensor(img_blurred)
		img_kernel=torch.FloatTensor(img_kernel.copy()).repeat(3, 1, 1)

		return img_kernel,img_blurred





#####################
# Saturated Dataset
#####################
class Low_light_Dataset(Dataset):
	def __init__(self, blurred,sharp,kernel,noise_level,gray_mode=False,additive_noise = False,random_flag = False):
		self.blurred = blurred
		self.sharp   =sharp
		self.kernel =kernel
		self.gray_mode = gray_mode
		self.additive_noise = additive_noise
		self.noise_level  =  noise_level 
		self.random_flag = random_flag
		#self.file_list = sorted([name for name in os.listdir(self.blurred) if os.path.isfile(os.path.join(self.blurred, name))])
		self.file_list = []
		self.sharp_num=len([name for name in os.listdir(self.sharp) if os.path.isfile(os.path.join(self.sharp, name))])
		# magic number
		self.kernel_num = 14 #len([name for name in os.listdir(self.kernel) if os.path.isfile(os.path.join(self.kernel, name))])
		for id in range(0,154):
			self.file_list.append( "{}_f{}.png".format(int(id/self.kernel_num)+1,int(id%self.kernel_num)+1) )
		# handle flag
		self.KERNEL_FLIP_FLAG = True
		self.BLURRED_FLAG = False 
		self.summary()

	def set_data_path(self,index):
		blurred_file = '%d_f%d.png'%(int(index/self.kernel_num)+1,int(index%self.kernel_num)+1)
		blurred  =os.path.join(self.blurred  ,blurred_file)
		sharp  =os.path.join(self.sharp  ,'%d_f%d_clear.png'%(int(index/self.kernel_num)+1,int(index%self.kernel_num)+1))
		kernel =os.path.join(self.kernel ,'%d.png'%(int(index%self.kernel_num)))#'{:06d}.jpg'
		return sharp,blurred,kernel

	def __len__(self):
		return self.sharp_num 

	def __getitem__(self, index):
		# handle path
		#blurred=os.path.join(self.blurred,'%d.jpg'%(index))
		sharp,blurred,kernel=self.set_data_path(index)
		img_sharp  = cv2.imread(sharp)/255.0
		img_kernel =cv2.imread(kernel,cv2.IMREAD_GRAYSCALE)
		if self.KERNEL_FLIP_FLAG:
			img_kernel=np.flip(img_kernel, [0,1])
			
		img_kernel =img_kernel/np.sum(img_kernel)
		print(blurred)
		img_blurred= cv2.imread(blurred)/255.0
		# Dimension swap
		img_blurred=np.transpose(img_blurred, (2,0,1))
		img_sharp  =np.transpose(img_sharp, (2,0,1))
		# pack to tensor
		img_kernel=torch.FloatTensor(img_kernel.copy()).repeat(3, 1, 1)
		img_sharp,img_blurred = torch.FloatTensor(img_sharp),torch.FloatTensor(img_blurred)
		if self.additive_noise:
			if self.random_flag:
				random_std = random.uniform(0, 1)*self.noise_level/255.0
				print("Add random noise std: {} * {}".format(random_std,self.noise_level/255.0))
				img_blurred=(img_blurred+np.random.normal(loc=0.0, scale=random_std,size=img_blurred.shape))#/255.0
			else:
				print("Add fixed noise std: {}".format(self.noise_level/255.0))
				img_blurred=(img_blurred+np.random.normal(loc=0.0, scale=self.noise_level/255.0,size=img_blurred.shape))#/255.0
		else:
			img_blurred=img_blurred
		if self.gray_mode:
			#c = index%3
			c = 0
			return img_kernel[c:c+1,:,:],img_sharp[c:c+1,:,:],img_blurred[c:c+1,:,:]
		else:
			return img_kernel,img_sharp,img_blurred

#####################
# Night Dataset
#####################
class Chen_Low_light_Dataset(Low_light_Dataset):
	def __init__(self, blurred,sharp,kernel,noise_level,gray_mode=True,additive_noise = False,random_flag = False):
		self.blurred = blurred
		self.sharp   =sharp
		self.kernel =kernel
		self.gray_mode = gray_mode
		self.additive_noise = additive_noise
		self.noise_level  =  noise_level 
		self.random_flag = random_flag
		#self.file_list = sorted([name for name in os.listdir(self.blurred) if os.path.isfile(os.path.join(self.blurred, name))])
		self.file_list = []
		self.sharp_num = len([name for name in os.listdir(self.sharp) if os.path.isfile(os.path.join(self.sharp, name))])
		self.kernel_num = len([name for name in os.listdir(self.kernel) if os.path.isfile(os.path.join(self.kernel, name))])
		# magic number
		for id in range(1,101):
			self.file_list.append( "{}.png".format(id) )
		# handle flag
		self.KERNEL_FLIP_FLAG = False
		self.BLURRED_FLAG = False 
		self.summary()

	def set_data_path(self,index):
		blurred_file = "{}.png".format(index+1)
		blurred = os.path.join(self.blurred,blurred_file)
		sharp   = os.path.join(self.sharp  ,blurred_file)
		kernel  = os.path.join(self.kernel ,blurred_file)
		return sharp,blurred,kernel

	def __len__(self):
		return len(self.file_list)


###########################
# Low-illumination Dataset 
###########################
class Pan_Low_light_Dataset(Low_light_Dataset):
	def __init__(self, blurred,sharp,kernel,noise_level,gray_mode=False):
		self.blurred = blurred
		self.sharp   =sharp
		self.kernel =kernel
		self.noise_level  =  noise_level 
		self.gray_mode = gray_mode
		self.additive_noise =False
		#self.file_list = sorted([name for name in os.listdir(self.blurred) if os.path.isfile(os.path.join(self.blurred, name))])
		self.file_list = []
		self.sharp_num = len([name for name in os.listdir(self.sharp) if os.path.isfile(os.path.join(self.sharp, name))])
		# magic number
		self.kernel_num = 8 #len([name for name in os.listdir(self.kernel) if os.path.isfile(os.path.join(self.kernel, name))])
		for id in range(0,48):
			self.file_list.append( "saturated_img{}_{}_blur.png".format(int(id/self.kernel_num)+1,int(id%self.kernel_num)+1) )
		# handle flag
		self.KERNEL_FLIP_FLAG = False
		self.BLURRED_FLAG = False 
		self.summary()

	def set_data_path(self,index):
		blurred_file = "saturated_img%d_%d_blur.png"%(int(index/self.kernel_num)+1,int(index%self.kernel_num)+1)
		blurred = os.path.join(self.blurred  ,blurred_file)
		sharp   = os.path.join(self.sharp  ,'saturated_img%d.png'%(int(index/self.kernel_num)+1))
		kernel  = os.path.join(self.kernel ,'ker0%d_truth.png'%(int(index%self.kernel_num)+1))#'{:06d}.jpg'
		return sharp,blurred,kernel

	def __len__(self):
		return len(self.file_list)



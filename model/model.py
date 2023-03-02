import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .util import *
import scipy.io as sio
from .boundary_wrap import opt_fft_size, wrap_boundary_RGB_torch
from .DRUNet_model.network_unet import  UNetRes as DRU_net

class DPSHS(nn.Module):
	def __init__(self, max_iter = 100,
			  		   _lambda = 2e-5,
					   rho = 0.1,
					   default_mode = "normal",
					   over_threshold = 5.0, 
					   hard_threshold = 0.9, 
					   ES_Threshold = 0.1, 
					   Monitor_FLAG = False):
		super(DPSHS, self).__init__()
		self.SET_GPU()
		self.build_filter()
		self.ES_Threshold = ES_Threshold
		self.Monitor_FLAG = Monitor_FLAG
		self.DEFAULT_MODE = default_mode
		self.OVER_THRESHOLD = over_threshold
		self.HARD_THRESHOLD = hard_threshold
		self.LAMBDA = torch.tensor(_lambda).view(1,1,1,1).to(self.device)
		self.RHO = torch.tensor(rho).view(1,1,1,1).to(self.device)
		self.MAX_ITER = max_iter
		self.ONES = torch.ones(1).to(self.device)
		self.ZEROS = torch.zeros(1).to(self.device)
		self.ALPHA = 1e2

	##############################
	# Utility Function
	##############################	
	def compute_diff(self, x, x_old):
		return torch.sqrt(torch.sum(torch.pow(x-x_old, 2))/(x.shape[-1]*x.shape[-2])).item()

	def myFFT(self,v):
		return torch.fft.fftn(v,dim=(-2,-1))

	def myIFFT(self,v):
		return torch.fft.ifftn(v,dim=(-2,-1)).real

	def FFT_conv(self,image,kernel):
		Fk = self.psf2otf(kernel,image.shape)
		Fx = torch.fft.fftn(image,dim=(-2,-1))
		kx = torch.fft.ifftn(Fk*Fx,dim=(-2,-1)).real	
		return kx 

	def RGBConv2D(self,image,kernel,padding="circular"):
		RGB_image=torch.zeros(image.shape).to(self.device)
		# pad last dim by (1, 1) and 2nd to last by (2, 2)
		p1d=(int(kernel.shape[3]/2),int(kernel.shape[3]/2),int(kernel.shape[2]/2),int(kernel.shape[2]/2))
		image= F.pad(image, p1d, padding)
		#image= F.pad(image, p1d, "reflect")
		for i in range(0,image.shape[1]):
			RGB_image[:,i:i+1,:,:]= F.conv2d(image[:,i:i+1,:,:], kernel)
		return RGB_image 

	def psf2otf(self,psf, shape):
		# psf: NxCxhxw
		# shape: [H,W]
		# otf: NxCxHxWx2
		# modified from USRNet/models/network_usrnet_v1.py 
		shape=(shape[2],shape[3])
		otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
		otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
		for axis, axis_size in enumerate(psf.shape[2:]):
			otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
		otf = torch.fft.fftn(otf, dim=(-2,-1))
		return otf

	def otf2psf(self,otf, psf_shape):
		# psf: NxCxhxw
		# shape: [H,W]
		# otf: NxCxHxWx2
		psf=torch.fft.ifftn(otf,dim=(-2,-1))
		for axis, axis_size in enumerate(psf_shape):
			psf = torch.roll(psf, int(axis_size / 2), dims=axis+2)
		psf=psf[:,:,0:psf_shape[0],0:psf_shape[1]] 
		return psf

	def SET_GPU(self,FLAG=None):
		if FLAG:
			self.device = FLAG
			print("Device mode:",self.device)
			return
		if torch.cuda.is_available():
			self.device='cuda'
		else:
			self.device='cpu'
		print("Device mode:",self.device)

	def pad_image(self, img_blurred, kernel, device, PAD_SCALE = 5, MOD_TIMES = 8):
		(N,C,H,W) = img_blurred.shape
		pad_shape =(H + PAD_SCALE*(kernel.shape[2]-1),W + PAD_SCALE*(kernel.shape[3]-1) ) 
		print("Padded  :",pad_shape)
		FFT_SHAPE = opt_fft_size(pad_shape)
		FFT_SHAPE = FFT_SHAPE[0]// MOD_TIMES*MOD_TIMES, FFT_SHAPE[1]// MOD_TIMES*MOD_TIMES
		img_blurred_pad = wrap_boundary_RGB_torch(img_blurred,FFT_SHAPE,device)
		return img_blurred_pad,H,W

	##############################
	# Generate matrix of Ax = b
	##############################	
	def gen_CG_b(self,v,mask):
		b = torch.conj(self.F_K)*self.myFFT(mask*v[:, 0, :, :, :]) \
			+ self.myFFT(v[:, 1, :, :, :]) \
			+ self.myFFT(v[:, 2, :, :, :])
		return b

	def NB_compute_Ax(self,x,mask):
		Fx = self.myFFT(x)
		M_Ax = mask*mask*self.myIFFT(Fx*self.F_K)
		AT_MT_M_Ax = self.myIFFT(torch.conj(self.F_K)*self.myFFT(M_Ax))
		return AT_MT_M_Ax+2.0*x

	def NB_quadratic_CG(self, x, v,mask, maxIt, tol=1e-5):
		b = self.gen_CG_b(v,mask)
		r = self.myIFFT(b) - self.NB_compute_Ax(x,mask)
		p = r
		rsold = torch.sum(r*r)
		# start CG
		# print("=> CG start")
		for iter in range(0, maxIt):
			Ap = self.NB_compute_Ax(p,mask)
			alpha = rsold/torch.sum(p*Ap)
			x = x+alpha*p
			r = r-alpha*Ap
			rsnew = torch.sum(r*r)
			# print(iter,rsnew)
			if torch.sqrt(rsnew) < tol:
				break
			p = r+rsnew/rsold*p
			rsold = rsnew
		return x

	def Prox_quad(self,v,kerenl,image):
		Nom = torch.conj(self.F_K)*self.myFFT(v[:,0,:,:,:]) \
	  		 +self.myFFT(v[:,1,:,:,:])
	
		Denom =   torch.conj(self.F_K)*self.F_K \
  			 	 +1.0
		return torch.fft.ifftn(Nom/Denom,dim=(-2,-1)).real

	##############################
	# Proximal Operators
	##############################
	def Prox_P(self,v,B,RHO):
		RHO = RHO+1e-8
		bias = -(1-RHO*v)/(2*RHO)
		var = torch.sqrt(bias*bias+B/RHO + 1e-8)
		assert(~var.isnan().any())
		return bias+var

	def Prox_I(self, v):
		return torch.clamp(v, min=0)

	def ProxF_PnP(self,_y,std,iter=0):
		_y = torch.cat((_y, std.float().repeat(_y.shape[0], 1, _y.shape[2], _y.shape[3])), dim=1)
		with torch.no_grad():
			y = self.CNN_denoiser(_y)
		return y

	##############################
	# Mask Related Functions
	##############################
	def update_M(self, x, k):
		Fk = self.psf2otf(k,x.shape)
		Fx = torch.fft.fftn(x,dim=(-2,-1))
		kx = torch.fft.ifftn(Fk*Fx,dim=(-2,-1)).real
		self.kx = kx
		M = torch.where(kx <=self.ONES,self.ONES,1.0/torch.max(kx,self.ONES*1e-6))
		if torch.isnan(M).int().sum():
			print("handle Nan Mask")
			assert(False)
		return M

	def image_mask(self,latent,CROSS_FLAG = True,TEMP_FLAG = True):
		############################################
		# decompose image to two parts
		# whole image S and  non-saturation region V 
		############################################
		N,C,H,W = latent.shape
		if CROSS_FLAG and C == 3:
			MASK = torch.logical_and(latent[:,0:1,:,:]>=self.HARD_THRESHOLD,latent[:,1:2,:,:]>=self.HARD_THRESHOLD)
			MASK = torch.logical_and(MASK,latent[:,2:3,:,:]>=self.HARD_THRESHOLD)
		else:
			MASK = latent>=self.HARD_THRESHOLD
		S_hard = torch.clamp(self.RGBConv2D((MASK.clone()).type(dtype=latent.dtype).to(self.device),self.dilate_filter),min=0.0,max=1.0)
		if TEMP_FLAG:
			S_mask = self.S_mask_0*S_hard
		else:
			S_mask = S_hard
		tmp = self.FFT_conv(S_mask,self.psf_dilate)
		V_mask = 1.0 - torch.minimum(tmp,self.ONES)
		blend_weights = self.FFT_conv(S_mask,self.smooth_filter)
		return S_mask,V_mask,blend_weights

 	################################
	# Generate filter
	################################
	def build_filter(self):
  		# generate pre-defined filter
		img_mat = sio.loadmat('model/pretrained/filter.mat')
		self.dilate_filter = torch.FloatTensor(img_mat["dilate_filter"]).unsqueeze(0).unsqueeze(0).to(self.device)
		self.smooth_filter = torch.FloatTensor(img_mat["smooth_filter"]).unsqueeze(0).unsqueeze(0).to(self.device)
		self.smooth_filter_T =  torch.flip(self.smooth_filter,[2, 3])
		# generate pre-trained denoiser
		self.CNN_denoiser = DRU_net(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
		self.CNN_denoiser.load_state_dict(torch.load("model/pretrained/drunet_color.pth"))
		self.CNN_denoiser.to(self.device)

	def forward(self,image,kernel,scale=0):
		#####################
		# Pad blurred image
		#####################
		image,H_ORI,W_ORI = self.pad_image(img_blurred = image, kernel = kernel, device = self.device)

		#####################
		# Initialization
		#####################		
		self.k_sz = kernel.shape[2]
		self.psf_dilate = (kernel!=0.0).type(kernel.dtype)
		kernel_flip = torch.flip(kernel, [2, 3])
		self.F_K = self.psf2otf(kernel,image.shape)
		self.F_K_flip = self.psf2otf(kernel_flip,image.shape)
		self.M = self.ONES	
		self.S_mask_0 = self.S_mask = torch.ones_like(image).type(torch.bool)
		self.S_mask_0,self.M_U,blend_weights = self.image_mask(image)

		#####################
		# ADMM Variable
		#####################		
		LAMBDA,RHO = self.LAMBDA, self.RHO 
		image_init,image_init_U = image,torch.zeros_like(image)
		x = image_init
		z = u = torch.stack([image_init]*2, axis=1)
		x_V = image_init_U
		z_V = u_V = torch.stack([image_init_U]*3, axis=1)

		if self.DEFAULT_MODE == "normal":
			extreme_flag = False
		elif self.DEFAULT_MODE == "over":
			extreme_flag = True
		else:
			print("QAQ => Non-defined initial mode")
			assert(False)

		# print("Default {} mode".format(self.DEFAULT_MODE))
		HALF_ITER = self.MAX_ITER//2
		SCALE = 1.0
		

		for k in range(0,self.MAX_ITER):
			x_old = x
			z_old = z
			u_old = u
			z_u = z-u
			x = self.Prox_quad(z_u,kernel,image/self.M)
			if extreme_flag:
				z_u_V = z_V-u_V
				x_V = self.NB_quadratic_CG(x=image if k == 0 or  (x_V == 0.0).all() else x_V,
									   v=z_u_V,
									   maxIt=20,
									   mask = self.M_U,
									   tol=1e-5)
				x = x_V+(x - x_V)*blend_weights
				v1_V = self.FFT_conv(x,kernel)*self.M_U+u_V[:,0,:,:,:]
				v3 = x + u_V[:,2,:,:,:]
				z1_V = self.Prox_P(v1_V,image*self.M_U,RHO)
				z3 = self.Prox_I(v3)
				self.S_mask,self.M_U,blend_weights = self.image_mask(x)

			max_pixel = torch.max(x)
			self.M = self.update_M(x,kernel)	
			# update v
			v1 = self.FFT_conv(x,kernel)+u[:,0,:,:,:]
			v2 = x + u[:,1,:,:,:]
			z1 = self.Prox_P(v1,image/self.M,RHO)
			v2 = torch.clamp(v2,min=0.0)
			z2 = self.ProxF_PnP(v2,torch.sqrt(max(SCALE,1.0)*LAMBDA/RHO),iter=k)
			z = torch.stack([z1,z2], axis=1)
			u_Kx = torch.stack([v1,v2], axis=1)
			# update multiplier
			u = u_Kx  - z
			if extreme_flag:
				z_V = torch.stack([z1_V,z2,z3], axis=1)
				u_Kx_V = torch.stack([v1_V,v2,v3], axis=1)
				u_V = u_Kx_V  - z_V
				
   

			if k>=1:
				x_diff, u_diff, z_diff = self.compute_diff(x_old,x),self.compute_diff(u_old,u),self.compute_diff(z_old,z)
				diff_sum = x_diff + u_diff + z_diff	
				#print("%d th iter %f %f %f %f"%(k,x_diff, u_diff, z_diff, diff_sum))
				#print("Stop Threshold {}".format(self.ES_Threshold/SCALE))
				if self.Monitor_FLAG:
					psnr_deblurred = torchPSNR(self.gt,torch.clamp(x[:,:,:self.H_ori,:self.W_ori],min=0.0,max=1.0))
					deblurred_ssim = piq.ssim(self.gt,torch.clamp(x[:,:,:self.H_ori,:self.W_ori],min=0.0,max=1.0),data_range =1.0)
					#print("PSNR:{} SSIM:{}".format(psnr_deblurred,deblurred_ssim))
				if diff_u_old < u_diff and diff_x_old < x_diff and diff_z_old < z_diff and diff_sum < self.ES_Threshold/SCALE and not self.extreme_flag:
					#print("Early stop at iter {}".format(k))
					break
				diff_u_old = u_diff
				diff_x_old = x_diff
				diff_z_old = z_diff
			else:
				diff_u_old = 1e8
				diff_x_old = 1e8
				diff_z_old = 1e8


			# adaptive mask 
			if max_pixel>=self.OVER_THRESHOLD and k<= HALF_ITER and not extreme_flag:
				print("===========================")
				print("switch to over-exposed mode")
				print("===========================")
				x,z,u = x_V,z_V[:,0:2,:,:,:],u_V[:,0:2,:,:,:,]
				extreme_flag = True

			self.extreme_flag = extreme_flag

			max_pixel = torch.max(x).item() 
			# print("Max pixel value",max_pixel)	
			SCALE = min(math.pow(2,max_pixel),self.ALPHA)

		return torch.clamp(x[:,:,:H_ORI,:W_ORI],min=0.0,max=1.0) 


class DPSHS_CG(DPSHS):
	##############################
	# Generate matrix of Ax = b
	##############################
	def gen_CG_b_2(self,v,mask):
		b = torch.conj(self.F_K)*self.myFFT(mask*v[:, 0, :, :, :]) \
			+ self.myFFT(v[:, 1, :, :, :])
		return b

	def NB_compute_Ax_2(self,x,mask):
		Fx = self.myFFT(x)
		M_Ax = mask*mask*self.myIFFT(Fx*self.F_K)
		AT_MT_M_Ax = self.myIFFT(torch.conj(self.F_K)*self.myFFT(M_Ax))
		return AT_MT_M_Ax+x

	def Prox_quad(self, x, v,mask, maxIt, tol=1e-5):
		b = self.gen_CG_b_2(v,mask)
		r = self.myIFFT(b) - self.NB_compute_Ax_2(x,mask)
		p = r
		rsold = torch.sum(r*r)
		# start CG
		# print("=> CG start")
		for iter in range(0, maxIt):
			Ap = self.NB_compute_Ax_2(p,mask)
			alpha = rsold/torch.sum(p*Ap)
			x = x+alpha*p
			r = r-alpha*Ap
			rsnew = torch.sum(r*r)
			#print(iter,rsnew)
			if torch.sqrt(rsnew) < tol:
				break
			p = r+rsnew/rsold*p
			rsold = rsnew
		return x

	def forward(self,image,kernel,scale=0):
		#####################
		# Pad blurred image
		#####################
		image,H_ORI,W_ORI = self.pad_image(img_blurred = image, kernel = kernel, device = self.device)

		#####################
		# Initialization
		#####################		
		self.k_sz = kernel.shape[2]
		self.psf_dilate = (kernel!=0.0).type(kernel.dtype)
		kernel_flip = torch.flip(kernel, [2, 3])
		self.F_K = self.psf2otf(kernel,image.shape)
		self.F_K_flip = self.psf2otf(kernel_flip,image.shape)
		self.M = self.ONES	
		self.S_mask_0 = self.S_mask = torch.ones_like(image).type(torch.bool)
		self.S_mask_0,self.M_U,blend_weights = self.image_mask(image)

		#####################
		# ADMM Variable
		#####################		
		LAMBDA,RHO = self.LAMBDA, self.RHO 
		image_init,image_init_U = image,torch.zeros_like(image)
		x = image_init
		z = u = torch.stack([image_init]*2, axis=1)
		x_V = image_init_U
		z_V = u_V = torch.stack([image_init_U]*3, axis=1)

		if self.DEFAULT_MODE == "normal":
			extreme_flag = False
		elif self.DEFAULT_MODE == "over":
			extreme_flag = True
		else:
			print("QAQ => Non-defined initial mode")
			assert(False)

		print("Default {} mode".format(self.DEFAULT_MODE))
		HALF_ITER = self.MAX_ITER//2
		SCALE = 1.0
		

		for k in range(0,self.MAX_ITER):
			x_old = x
			z_old = z
			u_old = u
			z_u = z-u
			#x = self.Prox_quad(z_u,kernel,image/self.M)
			x = self.Prox_quad(x=image if k == 0 or  (x == 0.0).all() else x,
							   v=z_u,
							   maxIt=20,
							   mask = self.M,
							   tol=1e-5)
			if extreme_flag:
				z_u_V = z_V-u_V
				x_V = self.NB_quadratic_CG(x=image if k == 0 or  (x_V == 0.0).all() else x_V,
									   v=z_u_V,
									   maxIt=20,
									   mask = self.M_U,
									   tol=1e-5)
				x = x_V+(x - x_V)*blend_weights
				v1_V = self.FFT_conv(x,kernel)*self.M_U+u_V[:,0,:,:,:]
				v3 = x + u_V[:,2,:,:,:]
				z1_V = self.Prox_P(v1_V,image*self.M_U,RHO)
				z3 = self.Prox_I(v3)
				self.S_mask,self.M_U,blend_weights = self.image_mask(x)

			max_pixel = torch.max(x)
			self.M = self.update_M(x,kernel)	
			# update v
			v1 = self.FFT_conv(x,kernel)*self.M+u[:,0,:,:,:]
			v2 = x + u[:,1,:,:,:]
			z1 = self.Prox_P(v1,image,RHO)
			v2 = torch.clamp(v2,min=0.0)
			z2 = self.ProxF_PnP(v2,torch.sqrt(max(SCALE,1.0)*LAMBDA/RHO),iter=k)
			z = torch.stack([z1,z2], axis=1)
			u_Kx = torch.stack([v1,v2], axis=1)
			# update multiplier
			u = u_Kx  - z
			if extreme_flag:
				z_V = torch.stack([z1_V,z2,z3], axis=1)
				u_Kx_V = torch.stack([v1_V,v2,v3], axis=1)
				u_V = u_Kx_V  - z_V
				
   

			if k>=1:
				x_diff, u_diff, z_diff = self.compute_diff(x_old,x),self.compute_diff(u_old,u),self.compute_diff(z_old,z)
				diff_sum = x_diff + u_diff + z_diff	
				print("%d th iter %f %f %f %f"%(k,x_diff, u_diff, z_diff, diff_sum))
				print("Stop Threshold {}".format(self.ES_Threshold/SCALE))
				if self.Monitor_FLAG:
					psnr_deblurred = torchPSNR(self.gt,torch.clamp(x[:,:,:self.H_ori,:self.W_ori],min=0.0,max=1.0))
					deblurred_ssim = piq.ssim(self.gt,torch.clamp(x[:,:,:self.H_ori,:self.W_ori],min=0.0,max=1.0),data_range =1.0)
					print("PSNR:{} SSIM:{}".format(psnr_deblurred,deblurred_ssim))
				if diff_u_old < u_diff and diff_x_old < x_diff and diff_z_old < z_diff and diff_sum < self.ES_Threshold/SCALE and not self.extreme_flag:
					print("Early stop at iter {}".format(k))
					break
				diff_u_old = u_diff
				diff_x_old = x_diff
				diff_z_old = z_diff
			else:
				diff_u_old = 1e8
				diff_x_old = 1e8
				diff_z_old = 1e8


			# adaptive mask 
			if max_pixel>=self.OVER_THRESHOLD and k<= HALF_ITER and not extreme_flag:
				print("===========================")
				print("switch to over-exposed mode")
				print("===========================")
				x,z,u = x_V,z_V[:,0:2,:,:,:],u_V[:,0:2,:,:,:,]
				extreme_flag = True

			self.extreme_flag = extreme_flag

			max_pixel = torch.max(x).item() 
			print("Max pixel value",max_pixel)	
			SCALE = min(math.pow(2,max_pixel),1e2)

		return torch.clamp(x[:,:,:H_ORI,:W_ORI],min=0.0,max=1.0) 
  
if __name__ == "__main__":
	Solver = DPSHS()
	print(Solver)
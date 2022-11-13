import argparse
import time
import torch
import numpy as np
import argparse
from data_loader.data_loaders import Low_light_DataLoader,Chen_Low_light_DataLoader,Pan_Low_light_DataLoader
import os
from tqdm import tqdm
from skimage import img_as_ubyte
import piq
from model.model import DPSHS
from model.util import MetricTracker, tensor_to_np_multiple, save_img_with_PSNR, matlabPSNR, torchPSNR, mkdirs
# Fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument("-dataset", "--dataset", type=str, default="Hu", help='')
parser.add_argument("-log", "--log", type=str, default="Compare_Hu/", help='')
parser.add_argument("-all_test", "--all_test", action='store_true', help='')
parser.add_argument("-local_path", "-local_path", type=str, default="./", help='')
args = parser.parse_args()
if __name__ == '__main__':
	# Environment setting
	if torch.cuda.is_available():
		device='cuda'
	else:
		device='cpu' 
	ALL_FLAG = args.all_test
	DATASET = args.dataset
	GRAY_MODE = False
	LOCAL_PATH = args.local_path
	IMG_DIR = os.path.join(args.local_path,args.log )
	SOLVER_LIST=[]

	if DATASET == "Hu":
		test_loader  = Low_light_DataLoader(
				blurred="{}/Benchmark/2014HU/blurry_image".format(LOCAL_PATH),
				sharp  = "{}/Benchmark/2014HU/clear_image".format(LOCAL_PATH),
				kernel =      "{}/Benchmark/2014HU/kernel".format(LOCAL_PATH),
				batch_size=1,
				shuffle=False,
				validation_split=0.0,
				training=False,
				num_workers=4) 
		if ALL_FLAG:
			print("all data mode: totoal {} image".format(test_loader.dataset.__len__()))
			SEED_LIST = range(0,test_loader.dataset.__len__())
		else:
   			SEED_LIST = [test_loader.get_data_set().file_list.index('10_f9.png'),
          				 test_loader.get_data_set().file_list.index('6_f10.png'),
				   		 test_loader.get_data_set().file_list.index('8_f13.png')]

	elif DATASET == "Chen":
		test_loader  = Chen_Low_light_DataLoader(
				blurred="{}/Benchmark/2021Chen/testset_public/blur_gray".format(LOCAL_PATH),
				sharp  =  "{}/Benchmark/2021Chen/testset_public/gt_gray".format(LOCAL_PATH),
				kernel =   "{}/Benchmark/2021Chen/testset_public/kernel".format(LOCAL_PATH),
				gray_mode=GRAY_MODE,
				batch_size=1,
				shuffle=False,
				validation_split=0.0,
				training=False,
				num_workers=4) 
		if ALL_FLAG:
			print("all data mode: totoal {} image".format(test_loader.dataset.__len__()))
			SEED_LIST = range(0,test_loader.dataset.__len__())
		else:
			SEED_LIST = [ test_loader.get_data_set().file_list.index('7.png'),
						  test_loader.get_data_set().file_list.index('10.png'),
						  test_loader.get_data_set().file_list.index('11.png')
        		]

	elif DATASET == "Pan":
		test_loader = Pan_Low_light_DataLoader(
			blurred="{}/Benchmark/2016Pan/low-illumination/blur_images".format(LOCAL_PATH),
			sharp  =  "{}/Benchmark/2016Pan/low-illumination/gt_images".format(LOCAL_PATH),
			kernel =     "{}/Benchmark/2016Pan/low-illumination/kernel".format(LOCAL_PATH),
			batch_size=1,
			shuffle=False,
			validation_split=0.0,
			training=False,
			num_workers=4) 
		if ALL_FLAG:
			print("all data mode: totoal {} image".format(test_loader.dataset.__len__()))
			SEED_LIST = range(0,test_loader.dataset.__len__())
		else:
			SEED_LIST=[
				   test_loader.get_data_set().file_list.index('saturated_img3_4_blur.png'),
				   test_loader.get_data_set().file_list.index('saturated_img6_4_blur.png')]

	solver = DPSHS(    max_iter = 100,
			  		   _lambda = 2e-5,
					   rho = 0.1,
					   default_mode = "normal",
					   over_threshold = 5.0, 
					   ES_Threshold = 0.1, 
					   Monitor_FLAG = False)
	SOLVER_LIST = [solver]

	psnr_tracker = MetricTracker("",writer = None) 
	time_tracker = MetricTracker("",writer = None) 
	ssim_tracker = MetricTracker("",writer = None) 
	for SEED in SEED_LIST:
		kernel,img,img_blurred = test_loader.dataset.__getitem__(index = SEED)
		kernel,img,img_blurred = kernel.unsqueeze(0).to(device),img.unsqueeze(0).to(device),img_blurred.unsqueeze(0).to(device)
		(N,C,H,W) = img_blurred.shape
		for i,solver in tqdm(enumerate(SOLVER_LIST)):	
			mode = type(solver).__name__
			filenames = test_loader.get_data_set().file_list[SEED][:-4]
			WO_LOG = os.path.join(IMG_DIR,"wo_log/")	
			#path setting
			mkdirs(WO_LOG)
			mkdirs(IMG_DIR)
			solver.eval()
			time_start = time.time()
			with torch.no_grad():
				deblurred_img=solver.forward(img_blurred,kernel)
			time_end = time.time()
			time_diff = time_end-time_start
			print("Total deblurring time:",time_diff) 
			###############
			# tensor2numpy
			###############
			deblurred_img_np = tensor_to_np_multiple(torch.clamp(deblurred_img,min=0.0,max=1.0))
			img_blurred_np = tensor_to_np_multiple(torch.clamp(img_blurred,min=0.0,max=1.0))
			img_np = tensor_to_np_multiple(img)
			kernel_np = tensor_to_np_multiple(kernel)

			##########################################
			# Scale Range of Pixels [0,1] ->[0,255]
			##########################################
			restored_img = img_as_ubyte(deblurred_img_np[0].copy())
			input_img = img_as_ubyte(img_blurred_np[0].copy())
			golden_img = img_as_ubyte(img_np[0].copy())
			kernel_img = img_as_ubyte(kernel_np[0].copy())
   
			###################
			# Compute SSIM/IQA
			###################
			deblurred_ssim = piq.ssim(img,torch.clamp(deblurred_img,min=0.0,max=1.0),data_range = 1.0)
			deblurred_psnr = matlabPSNR(restored_img, golden_img)

			###############
			# Save images
			###############	
			save_img_with_PSNR(os.path.join(WO_LOG, filenames+"_{}_wo.png".format(mode)),restored_img,golden_img,kernel_img,PSNR_flag=False)
			deblurred_log = os.path.join(IMG_DIR,  filenames+'_%s_deblurred.png'%(mode))
			deblurred_psnr,_ = save_img_with_PSNR(deblurred_log, restored_img.copy(),golden_img,kernel_img,WEIGHT_DICT={},PSNR = deblurred_psnr ,SSIM = deblurred_ssim)


			ssim_tracker.update(key = "{}_SSIM".format(mode),
								value = deblurred_ssim)
			psnr_tracker.update(key = "{}_PSNR".format(mode),
								value = deblurred_psnr)
			time_tracker.update(key = "{}_time".format(mode),
								value = time_diff)

			print("log image to {}".format(deblurred_log))


	#########################
	# Report Information
	#########################
	print("Log average SSIM")
	for key, value in ssim_tracker.result().items():
		print("    {:15s}: {}".format(str(key), value))

	print("Log average PSNR")
	for key, value in psnr_tracker.result().items():
		print("    {:15s}: {}".format(str(key), value))

	print("Log average time")
	for key, value in time_tracker.result().items():
		print("    {:15s}: {}".format(str(key), value))
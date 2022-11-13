import argparse
import time
import torch
import numpy as np
import argparse
from data_loader.data_loaders import Real_World_DataLoader
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
parser.add_argument("-log", "--log", type=str, default="Log/Compare_Real_World/", help='')
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
	LOCAL_PATH = args.local_path
	IMG_DIR = os.path.join(args.local_path,args.log )
	SOLVER_LIST=[]

	
	test_loader  = Real_World_DataLoader(
				blurred = "{}/Benchmark/Real_World/blurred".format(LOCAL_PATH),
				kernel="{}/Benchmark/Real_World/kernel".format(LOCAL_PATH),
				batch_size=1,
				shuffle=False,
				validation_split=0.0,
				training=False,
				num_workers=4) 
	SEED_LIST = range(0,test_loader.dataset.__len__())
	if ALL_FLAG:
		print("all data mode: totoal {} image".format(test_loader.dataset.__len__()))
		SEED_LIST = range(0,test_loader.dataset.__len__())
	else:
		SEED_LIST = range(0,test_loader.dataset.__len__())




	solver = DPSHS(    max_iter = 100,
			  		   _lambda = 2e-5,
					   rho = 0.1,
					   default_mode = "normal",
					   over_threshold = 5.0, 
					   ES_Threshold = 0.1, 
					   Monitor_FLAG = False)
	SOLVER_LIST = [solver]
	time_tracker = MetricTracker("",writer = None) 
	for SEED in SEED_LIST:
		kernel,img_blurred = test_loader.dataset.__getitem__(index = SEED)
		kernel,img_blurred = kernel.unsqueeze(0).to(device),img_blurred.unsqueeze(0).to(device)
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
			kernel_np = tensor_to_np_multiple(kernel)

			##########################################
			# Scale Range of Pixels [0,1] ->[0,255]
			##########################################
			restored_img = img_as_ubyte(deblurred_img_np[0].copy())
			input_img = img_as_ubyte(img_blurred_np[0].copy())
			kernel_img = img_as_ubyte(kernel_np[0].copy())

			###############
			# Save images
			###############	
			save_img_with_PSNR(os.path.join(WO_LOG, filenames+"_{}_wo.png".format(mode)),restored_img,None,kernel_img,PSNR_flag=False)


			time_tracker.update(key = "{}_time".format(mode),
								value = time_diff)



	#########################
	# Report Information
	#########################

	print("Log average time")
	for key, value in time_tracker.result().items():
		print("    {:15s}: {}".format(str(key), value))
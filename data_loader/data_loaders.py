from os import scandir
from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.Blurreddata import Dataset, Pan_Low_light_Dataset,Chen_Low_light_Dataset, Low_light_Dataset
from data_loader.Blurreddata import Real_World_Dataset
import torchvision

class DataLoader(BaseDataLoader):
    def __init__(self,sharp,kernel,noise_level, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = Dataset(sharp,kernel,noise_level)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
    @property
    def get_batch_size(self):
        print("Batch size",self.b_sz)
        return self.b_sz 

'''
Blind Deblur Benchmark Dataset
'''
#############
# Real World
#############
class Real_World_DataLoader(BaseDataLoader):
    def __init__(self,blurred,sharp,kernel,noise_level, batch_size,gray_mode=False,shuffle=True, validation_split=0.0, num_workers=1, training=True,additive_noise = False,random_flag = False):
        self.dataset = Real_World_Dataset( sharp,kernel,noise_level)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

#############
# 2021 CVPR
#############
class Chen_Low_light_DataLoader(BaseDataLoader):
    def __init__(self,blurred,sharp,kernel,noise_level, batch_size,gray_mode=False,shuffle=True, validation_split=0.0, num_workers=1, training=True,additive_noise = False,random_flag = False):
        self.dataset = Chen_Low_light_Dataset(blurred,sharp,kernel,noise_level,gray_mode,additive_noise = additive_noise,random_flag = random_flag)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

#############
# 2016 TPAMI
#############
class Pan_Low_light_DataLoader(BaseDataLoader):
    def __init__(self,blurred,sharp,kernel,noise_level, batch_size,gray_mode=False,shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = Pan_Low_light_Dataset(blurred,sharp,kernel,noise_level,gray_mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class Pan_Low_light_noise_DataLoader(BaseDataLoader):
    def __init__(self,blurred,sharp,kernel,noise_level, batch_size,gray_mode=False,shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = Pan_Low_light_noise_Dataset(blurred,sharp,kernel,noise_level,gray_mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

#############
# 2016 CVPR
#############
class Lai_Low_light_DataLoader(BaseDataLoader):
    def __init__(self,blurred,sharp,kernel,noise_level, batch_size,gray_mode=False,shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = Lai_Low_light_Dataset(blurred,sharp,kernel,noise_level,gray_mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class Low_light_DataLoader(BaseDataLoader):
    def __init__(self,blurred,sharp,kernel,noise_level, batch_size,gray_mode=False,shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = Low_light_Dataset(blurred,sharp,kernel,noise_level,gray_mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class BD_2021Chen_Low_light_DataLoader(BaseDataLoader):
    def __init__(self,blurred,sharp,kernel,noise_level, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = BD_2021Chen_Low_light_Dataset(blurred,sharp,kernel,noise_level)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class BD_2021Pan_Low_light_DataLoader(BaseDataLoader):
    def __init__(self,blurred,sharp,kernel,noise_level, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = BD_2021Pan_Low_light_Dataset(blurred,sharp,kernel,noise_level)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class Levin_DataLoader(BaseDataLoader):
    def __init__(self,mat_path,noise_level, batch_size,gray_mode=False, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = Levin_Gray_Dataset(mat_path,noise_level)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class Kohler_DataLoader(BaseDataLoader):
    def __init__(self,sharp,blurred,noise_level, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = Kohler_Dataset(sharp,blurred,noise_level)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class Cho_DataLoader(BaseDataLoader):
    def __init__(self,data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = Cho_Dataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

'''
Synthetic Blind Deblur Benchmark Dataset
'''

class Synthetic_low_light_loader(BaseDataLoader):
    def __init__(self,blurred,sharp,kernel,noise_level,gray_mode, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = Low_Light_Synthetic(blurred,sharp,kernel,noise_level,gray_mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class My_Synthetic_low_light_loader(BaseDataLoader):
    def __init__(self,sharp,kernel,noise_level,mode, batch_size,dataset_src,gray_mode=False, shuffle=True,crop_flag= False,clip_flag=True,scale_factor = 9, validation_split=0.0, num_workers=1,random_flag=False):
        self.dataset = My_Low_Light_Synthetic(sharp,kernel,noise_level,mode,dataset_src=dataset_src,gray_mode=gray_mode,crop_flag=crop_flag,scale_factor=scale_factor,clip_flag=clip_flag,random_flag=random_flag)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


'''
Self low light Dataset
'''
class Self_low_light_loader(BaseDataLoader):
    def __init__(self,blurred, sharp, kernel,noise_level,sample_times =50,Patch_size = 256,gray_mode=False, batch_size =1, shuffle=True, validation_split=0.0, num_workers=1,training=True):
        self.dataset = Self_Low_light_Dataset(blurred, sharp, kernel, noise_level,sample_times = sample_times,Patch_size = Patch_size,gray_mode=gray_mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == "__main__":
    test_loader = Self_low_light_loader(
		blurred="/home/r09021/unfolding_research/Motion_deblur/Benchmark/2014HU/blurry_image/1_f1.png",
		sharp="/home/r09021/unfolding_research/Motion_deblur/Benchmark/2014HU/clear_image/1_f1_clear.png",
		kernel="/home/r09021/unfolding_research/Motion_deblur/Benchmark/2014HU/kernel/0.png",
		noise_level=1, #1% gaussian noise 
		batch_size=1,
		shuffle=False,
		validation_split=0.0,
		training=False,
		num_workers=4
	) 
    #for seed in range(0,2):
    sharp,blurred,kerenl = test_loader.dataset.__getitem__(index = 0)
    print(sharp)
    sharp,blurred,kerenl = test_loader.dataset.__getitem__(index = 0)
    print(sharp)
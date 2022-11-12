from base import BaseDataLoader
from data_loader.Blurreddata import Pan_Low_light_Dataset,Chen_Low_light_Dataset, Low_light_Dataset,Real_World_Dataset

#############
# 2014 Hu
#############
class Low_light_DataLoader(BaseDataLoader):
    def __init__(self,blurred,sharp,kernel,noise_level, batch_size,gray_mode=False,shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = Low_light_Dataset(blurred,sharp,kernel,noise_level,gray_mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

#############
# 2021 Chen
#############
class Chen_Low_light_DataLoader(BaseDataLoader):
    def __init__(self,blurred,sharp,kernel,noise_level, batch_size,gray_mode=False,shuffle=True, validation_split=0.0, num_workers=1, training=True,additive_noise = False,random_flag = False):
        self.dataset = Chen_Low_light_Dataset(blurred,sharp,kernel,noise_level,gray_mode,additive_noise = additive_noise,random_flag = random_flag)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

#############
# 2016 Pan
#############
class Pan_Low_light_DataLoader(BaseDataLoader):
    def __init__(self,blurred,sharp,kernel,noise_level, batch_size,gray_mode=False,shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = Pan_Low_light_Dataset(blurred,sharp,kernel,noise_level,gray_mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

#############
# Real World
#############
class Real_World_DataLoader(BaseDataLoader):
    def __init__(self,blurred,sharp,kernel,noise_level, batch_size,gray_mode=False,shuffle=True, validation_split=0.0, num_workers=1, training=True,additive_noise = False,random_flag = False):
        self.dataset = Real_World_Dataset( sharp,kernel,noise_level)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


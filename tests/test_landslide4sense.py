import sys
import os
import pdb
from Dataset4EO.datasets import landslide4sense
from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
#from torchdata.dataloader2 import MultiProcessingReadingService
import time
from tqdm import tqdm

datasets_dir = './'

if __name__ == '__main__':
    dp = landslide4sense.Landslide4Sense(datasets_dir, split='train')
    #dp = landslide4sense.Landslide4Sense(datasets_dir, split='val')
    #mprs = MultiProcessingReadingService(num_workers=4, pin_memory=True, prefetch_factor=2,  persistent_workers=True)

    #data_loader = iter(DataLoader(dp.batch(4), reading_service=mprs))
    data_loader = DataLoader2(dp.shuffle(), batch_size=4, num_workers=4, shuffle=True, drop_last=True)
    for epoch in range(2):
        t1 = time.time()
        for it in tqdm(data_loader):
            # print(it['img_name'])
            pass
        t2 = time.time()
        print(t2-t1)



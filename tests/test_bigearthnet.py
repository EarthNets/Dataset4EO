import sys
import os
import pdb
from Dataset4EO.datasets import BigEarthNet
from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm

datasets_dir = '/mnt/d/codes/datasets/BigEarthNet/BigEarthNet-S2-v1.0/'
from torchdata.dataloader2 import MultiProcessingReadingService

if __name__ == '__main__':
    dp = BigEarthNet(datasets_dir, split='train')
    data_loader = DataLoader2(dp.shuffle(), batch_size=4, num_workers=4, shuffle=True,
                              drop_last=True)
    for epoch in range(5):
        t1 = time.time()
        for it in tqdm(data_loader):
            pass
            print(it)
        t2 = time.time()
        print('loading time: {}'.format(t2-t1))



import sys
import os
import pdb
from Dataset4EO.datasets import Inria
from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm

datasets_dir = '../../Datasets/Dataset4EO'
from torchdata.dataloader2 import MultiProcessingReadingService

if __name__ == '__main__':
    dp = Inria(datasets_dir, split='train_val')
    data_loader = DataLoader2(dp.shuffle(), batch_size=1, num_workers=0, shuffle=True,
                              drop_last=True)
    for epoch in range(1):
        t1 = time.time()
        for it in tqdm(data_loader):
            pass
            print(it)
        t2 = time.time()
        print('loading time: {}'.format(t2-t1))



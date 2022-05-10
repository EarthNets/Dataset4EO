import sys
import os
import pdb
from Dataset4EO.datasets import landslide4sense
from torch.utils.data import DataLoader2
import time
from tqdm import tqdm

datasets_dir = './'

if __name__ == '__main__':
    dp = landslide4sense.Landslide4Sense(datasets_dir, split='train')
    # dp = landslide4sense.Landslide4Sense(datasets_dir, split='val')

    data_loader = DataLoader2(dp.shuffle(), batch_size=4, num_workers=4, shuffle=True)
    for epoch in range(1):
        t1 = time.time()
        for it in tqdm(data_loader):
            # print(it['img_name'])
            pass
        t2 = time.time()
        print(t2-t1)



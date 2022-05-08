from Dataset4EO import datasets
import torch
from typing import Any, Dict, List, Tuple, Union, BinaryIO
from Dataset4EO.features import EncodedData
from Dataset4EO.utils._internal import fromfile, ReadOnlyTensorBuffer, parse_h5py, bytefromfile
import platform
import io
import mmap
import contextlib
import os
import numpy as np

#tr = datasets.landcover_sen2.LandCoverSen2('./')
from Dataset4EO.datasets.utils import Dataset, HttpResource, OnlineResource
import tempfile


tr = datasets.landcover_sen2.LandCoverSen2('./', split='train')

from torch.utils.data import DataLoader2
datas = DataLoader2(tr, batch_size=4, num_workers=4, shuffle=True)
ii = 0

for data in iter(tr):
    ii+=1
    print(data['img'][()])


print('---------------------------------')
print(ii)



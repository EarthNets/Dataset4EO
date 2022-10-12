import os
import tarfile
import enum
import functools
import pathlib
import glob
from tqdm import tqdm
import h5py
import torch
from typing import Any, Dict, List, Optional, Tuple, BinaryIO, cast, Union
from xml.etree import ElementTree
from torch.utils.data import DataLoader2
from Dataset4EO import transforms
import pdb
import numpy as np

from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Filter,
    Demultiplexer,
    IterKeyZipper,
    LineReader,
)

from torchdata.datapipes.map import SequenceWrapper

from Dataset4EO.datasets.utils import OnlineResource, HttpResource, Dataset
from Dataset4EO.datasets.utils._internal import (
    path_accessor,
    getitem,
    INFINITE_BUFFER_SIZE,
    path_comparator,
    hint_sharding,
    hint_shuffling,
    read_categories_file,
)
from Dataset4EO.features import BoundingBox, Label, EncodedImage

from .._api import register_dataset, register_info

NAME = "BigEarthNet"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))


@register_dataset(NAME)
class BigEarthNet(Dataset):
    """
    - **homepage**: https://bigearth.net/
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:


        self._split = self._verify_str_arg(split, "split", ("train", "val", "test"))
        self.root = root
        self._categories = _info()["categories"]
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _prepare_sample(self, idx):
        data_info = self.data_item[idx]
        
        return data_info

    def _prepare_list(self):
        self.data_item = []
        with open(os.path.join(self.root, '19_labels_{}.csv'.format(self._split))) as f:
            samples = [x.strip().split(',') for x in f.readlines()]
            
            for sample in samples:
                filename = sample[0] + '.png'
                gt_label = sample[1:]
                target = np.zeros(19, dtype=np.uint8)
                for id in gt_label:
                    target[int(id)] = 1
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(target, dtype=np.uint8) # binary
                self.data_item.append(info)
                
               

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        self._decompress_dir()
        self._prepare_list()

        dp = SequenceWrapper(range(self.__len__()))

        ndp = Mapper(dp, self._prepare_sample)
        ndp = hint_shuffling(ndp)
        ndp = hint_sharding(ndp)

        return ndp

    def __len__(self) -> int:

        return len(self.data_item)

if __name__ == '__main__':
    dp = BigEarthNet('./')

import os
import tarfile
import enum
import functools
import pathlib
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
    Zipper
)

from torchdata.datapipes.map import SequenceWrapper

from Dataset4EO.datasets.utils import OnlineResource, HttpResource, Dataset, ManualDownloadResource
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

NAME = "dior"
_TRAIN_LEN = 5862
_VAL_LEN = 5863
_TEST_LEN = 11738


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))


@register_dataset(NAME)
class DIOR(Dataset):
    """
    - **paper link**: https://arxiv.org/abs/1909.00133?context=cs.LG.html
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:

        assert split == 'train' or split == 'val' or split == 'test'
        self._split = split
        self.root = root
        self._categories = _info()["categories"]
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _CHECKSUMS = {
        "all": ("https://syncandshare.lrz.de/dl/firng4uPNLx9FLPxdBFmP5X/DIOR.zip",
                "7a674c72a23ccca91eaf46c7cebdaa2bb9dbd6056af4ecf5c213f32f3d42ad92"),
    }

    def get_classes(self):
        return self._categories

    def _resources(self) -> List[OnlineResource]:
        resource = HttpResource(
            url = self._CHECKSUMS['all'][0],
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['all'][1]
        )

        return [resource]

    def _prepare_sample(self, data):

        image_data, ann_data = data[1]
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        img_info = {'filename':image_path,
                    'img_id': image_path.split('/')[-1].split('.')[0],
                    'ann':{'ann_path': ann_path}}

        return img_info

    def _select_split(self, data):
        path = pathlib.Path(data[0])
        return path.parents[0].name == 'JPEGImages-{}'.format(self._split)

    def _classify_archive(self, data):
        path = pathlib.Path(data[0])
        if path.name.endswith(f'{self._split}.txt'):
            return 0
        elif path.name.endswith('jpg'):
            return 1
        elif path.name.endswith('xml') and path.parent.name == 'Horizontal Bounding Boxes':
            return 2
        elif path.name.endswith('xml') and path.parent.name == 'Oriented Bounding Boxes':
            return 3
        else:
            return None

    def _split_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        return data[1].decode('UTF-8')

    def _anns_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])
        return path.name.split('.')[0]

    def _images_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])
        return path.name.split('.')[0]

    def _dp_key_fn(self, data):
        path = pathlib.Path(data[0][0])
        return path.name.split('.')[0]


    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        split_dp, img_dp, ann_dp_h, ann_dp_o = Demultiplexer(
            resource_dps[0], 4, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )
        # img_dp = Filter(img_dp, self._select_split)
        # dp = Zipper(img_dp, ann_dp_h, ann_dp_o)
        dp = IterKeyZipper(
            img_dp, ann_dp_h,
            key_fn=self._images_key_fn,
            ref_key_fn=self._anns_key_fn,
            buffer_size=INFINITE_BUFFER_SIZE,
            keep_key=False
        )

        split_dp = LineReader(split_dp)
        dp = IterKeyZipper(
            split_dp, dp,
            key_fn=self._split_key_fn,
            ref_key_fn=self._dp_key_fn,
            buffer_size=INFINITE_BUFFER_SIZE,
            keep_key=False
        )

        # dp = Zipper(img_dp, ann_dp)

        ndp = Mapper(dp, self._prepare_sample)
        ndp = hint_shuffling(ndp)
        ndp = hint_sharding(ndp)

        return ndp

    def __len__(self) -> int:
        return {
            'train': _TRAIN_LEN,
            'val': _VAL_LEN,
            'test': _TEST_LEN
        }[self._split]

if __name__ == '__main__':
    dp = Landslide4Sense('./')

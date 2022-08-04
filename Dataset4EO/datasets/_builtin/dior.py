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
_TRAVAL_LEN = 11725
_VAL_LEN = 11725


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
        split: str = "trainval",
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:

        assert split == 'trainval' or split == 'test'
        self._split = split
        self.root = root
        self._categories = _info()["categories"]
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _CHECKSUMS = {
        "all": ("https://syncandshare.lrz.de/dl/firng4uPNLx9FLPxdBFmP5X/DIOR.zip",
                "b0609d80ca827a5ec854c3b0c3e63dce467e9ebbaec32e2420b71ea8e821c1a6"),
    }

    def _resources(self) -> List[OnlineResource]:
        resource = HttpResource(
            url = self._CHECKSUMS['all'][0],
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['all'][1]
        )

        return [resource]

    def _prepare_sample(self, data):

        image_data, ann_data_h, ann_data_o = data
        image_path, image_buffer = image_data
        ann_path_h, ann_buffer = ann_data_h
        ann_path_o, ann_buffer = ann_data_o

        img_info = {'filename':image_path,
                    'ann':{'ann_path_horizontal': ann_path_h,
                           'ann_path_oriented': ann_path_o}}

        return img_info

    def _select_split(self, data):
        path = pathlib.Path(data[0])
        return path.parents[0].name == 'JPEGImages-{}'.format(self._split)

    def _classify_archive(self, data):
        path = pathlib.Path(data[0])
        if path.name.endswith('jpg'):
            return 0
        elif path.name.endswith('xml') and path.parent.name == 'Horizontal Bounding Boxes':
            return 1
        elif path.name.endswith('xml') and path.parent.name == 'Oriented Bounding Boxes':
            return 2
        else:
            return None

    def _images_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])
        return path.name.split('.')[0]

    def _anns_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])
        return path.name.split('.')[0]

    def _images_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])
        return path.name.split('.')[0]


    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        img_dp, ann_dp_h, ann_dp_o = Demultiplexer(
            resource_dps[0], 3, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )
        img_dp = Filter(img_dp, self._select_split)

        # dp = IterKeyZipper(
        #     img_dp, ann_dp_h,
        #     key_fn=self._images_key_fn,
        #     ref_key_fn=self._anns_key_fn,
        #     buffer_size=INFINITE_BUFFER_SIZE,
        #     keep_key=True
        # )
        dp = Zipper(img_dp, ann_dp_h, ann_dp_o)

        ndp = Mapper(dp, self._prepare_sample)
        ndp = hint_shuffling(ndp)
        ndp = hint_sharding(ndp)

        return ndp

    def __len__(self) -> int:
        return {
            'trainval': _TRAVAL_LEN,
            'val': _VAL_LEN
        }[self._split]

if __name__ == '__main__':
    dp = Landslide4Sense('./')

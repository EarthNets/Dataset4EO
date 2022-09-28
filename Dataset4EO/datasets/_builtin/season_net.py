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
    Zipper,
    ZipperLongest,
    Concater,
    FileOpener
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

NAME = "season_net"
_TRAIN = 1232007
_VAL = 176117
_TEST = 351706
_TRAIN_SPRING = 322066
_TRAIN_SUMMER = 337015
_TRAIN_FALL = 350073
_TRAIN_WINTER = 153218
_TRAIN_SNOW = 69635
_VAL_SPRING = 46027
_VAL_SUMMER = 48203
_VAL_FALL = 50023
_VAL_WINTER = 21918
_VAL_SNOW = 9946
_TEST_SPRING = 92011
_TEST_SUMMER = 96238
_TEST_FALL = 100053
_TEST_WINTER = 43527
_TEST_SNOW = 19877


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class SeasonNetResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        """
        # Download SeasonNet data manually:
        wget https://zenodo.org/record/6979994/files/fall.zip
        wget https://zenodo.org/record/6979994/files/meta.csv
        wget https://zenodo.org/record/6979994/files/snow.zip
        wget https://zenodo.org/record/6979994/files/splits.zip
        wget https://zenodo.org/record/6979994/files/spring.zip
        wget https://zenodo.org/record/6979994/files/summer.zip
        wget https://zenodo.org/record/6979994/files/winter.zip
        """
        super().__init__('For data download, please refer to https://zenodo.org/record/6979994',
                         **kwargs)

@register_dataset(NAME)
class SeasonNet(Dataset):
    """
    - **dataset link**: https://zenodo.org/record/6979994
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split = "train_rural",
        season = "all",
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:

        # assert split in ['train_rural', 'train_urban', 'val_rural', 'val_urban', 'test_rural', 'test_urban']
        assert split in ['train', 'val', 'test']
        assert season in ['spring', 'summer', 'fall', 'winter', 'snow', 'all']

        self._split = split
        self.season = season

        self.root = root
        self._categories = _info()["categories"]
        self.CLASSES = self._categories
        self.PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128]]
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _CHECKSUMS = {
        'spring.zip': '28361db226817eadbea6fff378899d606d010ab796fa25b53c89923873b54b8e',
        'summer.zip': 'ecf1cd4b53d4627eee4b7a2548a01836bb7b64f3f881e24041fa310e378d43c0',
        'fall.zip': '86fb7c96521c9d1e2ef28fb9b2d32f80f10e6da922b80458478960025ff5073e',
        'winter.zip': '8c7f059f1583b2ce748c154743f7896bfe16e7f02f47a910131b75ccc31beb65',
        'snow.zip': '766d7f6d526008e1f2badd96d70ec81801fae473483e96dba24fc25015c01239',
        'meta.csv': '905f344353769fc5722930413a803e885b8928b2a9af220963a5aa0e30457d8c',
        'splits.zip': 'b938bd599569c21281af34485fbd66cf08536265c74ff41e745c4698a61738bb',
    }

    def get_classes(self):
        return self._categories

    def _resources(self) -> List[OnlineResource]:
        spring_resource = SeasonNetResource(
            file_name = 'spring.zip',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['spring.zip']
        )

        summer_resource = SeasonNetResource(
            file_name = 'summer.zip',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['summer.zip']
        )

        fall_resource = SeasonNetResource(
            file_name = 'fall.zip',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['fall.zip']
        )

        winter_resource = SeasonNetResource(
            file_name = 'winter.zip',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['winter.zip']
        )

        snow_resource = SeasonNetResource(
            file_name = 'snow.zip',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['snow.zip']
        )

        meta_resource = SeasonNetResource(
            file_name = 'meta.csv',
            preprocess = None,
            sha256 = self._CHECKSUMS['meta.csv']
        )

        splits_resource = SeasonNetResource(
            file_name = 'splits.zip',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['splits.zip']
        )

        return [spring_resource, summer_resource, fall_resource,
                winter_resource, snow_resource, meta_resource,
                splits_resource]

    def _idx2meta(self, data):
        return self.meta_list[int(data[0])+1]

    def _get_img_ann(self, data):
        folder_path = data[-1]
        season_name = folder_path.split('/')[0]
        file_name = folder_path.split('/')[-1]

        img_path = os.path.join(self.root, season_name, folder_path, f'{file_name}_10m_RGB.tif')
        ann_path = os.path.join(self.root, season_name, folder_path, f'{file_name}_labels.tif')

        img_info = dict(
            filename = img_path,
            img_id = file_name,
            ann = {'seg_map': ann_path}
        )

        return img_info

    def _filter_split(self, data):
        meta = eval(f'self.{self._split}_meta')
        path = pathlib.Path(data[0])
        folder_name = path.parent.name

        return folder_name in meta

    def _filter_season(self, data):
        season = data[1].lower()

        return season == self.season or self.season == 'all'

    def _classify_archive(self, data):
        path = pathlib.Path(data[0])
        file_type = path.name.split('_')[-1].split('.tif')[0]
        if file_type == 'RGB':
            return 0
        elif file_type == 'labels':
            return 1

    def _prepare_sample(self, data):
        image_data, ann_data = data
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        img_info = dict({'filename': image_path, 'ann': dict({'seg_map': ann_path})})

        return img_info



    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        spring_dp, summer_dp, fall_dp, winter_dp, snow_dp, meta_dp, split_dp = resource_dps

        meta_path = iter(meta_dp).__next__()[0]
        meta_dp = FileOpener([meta_path], mode='r')
        meta_dp = meta_dp.parse_csv(delimiter=',')
        meta_list = list(iter(meta_dp))

        self.meta_list = meta_list

        test_dp, train_dp, val_dp = split_dp
        train_dp = FileOpener([iter(train_dp).__next__()], mode='r').parse_csv()
        val_dp = FileOpener([iter(val_dp).__next__()], mode='r').parse_csv()
        test_dp = FileOpener([iter(test_dp).__next__()], mode='r').parse_csv()

        # train_list = list(iter(train_dp))
        # val_list = list(iter(val_dp))
        # test_list = list(iter(test_dp))

        # train_meta = {}
        # for idx in train_list:
        #     meta = self.meta_list[int(idx[0]) + 1]
        #     img_name = meta[-1].split('/')[-1]
        #     train_meta[img_name] = meta

        # val_meta = {}
        # for idx in val_list:
        #     meta = self.meta_list[int(idx[0]) + 1]
        #     img_name = meta[-1].split('/')[-1]
        #     val_meta[img_name] = meta

        # test_meta = {}
        # for idx in test_list:
        #     meta = self.meta_list[int(idx[0]) + 1]
        #     img_name = meta[-1].split('/')[-1]
        #     test_meta[img_name] = meta

        # self.train_meta = train_meta
        # self.val_meta = val_meta
        # self.test_meta = test_meta

        train_dp = Mapper(train_dp, self._idx2meta)
        val_dp = Mapper(val_dp, self._idx2meta)
        test_dp = Mapper(test_dp, self._idx2meta)

        dp = eval(f'{self._split}_dp')
        dp = Filter(dp, self._filter_season)
        # temp = iter(dp)
        # temp2 = temp.__next__()

        # if self.season == 'all':
        #     season_dp = Concater(spring_dp, summer_dp, fall_dp, winter_dp, snow_dp)
        # else:
        #     season_dp = eval(f'{self.season}_dp')

        # dp = Filter(season_dp, self._filter_split)

        # img_dp, ann_dp = Demultiplexer(
        #     dp, 2, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        # )

        # dp = Zipper(img_dp, ann_dp)

        dp = Mapper(dp, self._get_img_ann)
        # dp = Mapper(dp, self._prepare_sample)
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)

        return dp

    def __len__(self) -> int:
        if self.season == 'all':
            length = eval(f'_{self._split.upper()}')
        else:
            length = eval(f'_{self._split.upper()}_{self.season.upper()}')

        return length

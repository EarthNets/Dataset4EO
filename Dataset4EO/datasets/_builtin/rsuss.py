import os
import tarfile
import enum
import functools
from tqdm import tqdm
import h5py
import torch
from typing import Any, Dict, List, Optional, Tuple, BinaryIO, cast, Union
from xml.etree import ElementTree
from Dataset4EO import transforms
import pathlib
import pdb
import numpy as np

from torchdata.datapipes.iter import FileLister, FileOpener, StreamReader
from PIL import Image

from torchdata.datapipes.iter import Mapper

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

NAME = "rsuss"
FNAME = "RSUSS"
_TRAIN_LEN = 5137
_VAL_LEN = 1059
_TEST_LEN = 3144


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))


@register_dataset(NAME)
class RSUSS(Dataset):
    """
    - **homepage**: https://www.iarai.ac.at/rsbenchmark4uss/
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        mode: str = "unsupervised",
        skip_integrity_check: bool = False,
    ) -> None:

        self._split = self._verify_str_arg(split, "split", ("train", "val", "test"))
        self.root = root
        self._categories = _info()["categories"]
        self.mode = mode

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _TRAIN_VAL_ARCHIVES = {
        "trainval": ("landslide4sense.tar", "c7f6678d50c7003eba47b3cace8053c9bfa6b4692cd1630fe2d6b7bec11ccc77"),
    }

    def decompress_integrity_check(self, decom_dir):
        train_img_dir = os.path.join(decom_dir, 'train', 'img')
        train_mask_dir = os.path.join(decom_dir, 'train', 'mask')
        val_img_dir = os.path.join(decom_dir, 'val', 'img')

        if not os.path.exists(train_img_dir) or not os.path.exists(train_mask_dir) or not os.path.exists(val_img_dir):
            return False

        num_train_img = len(os.listdir(train_img_dir))
        num_train_mask = len(os.listdir(train_mask_dir))
        num_val_img = len(os.listdir(val_img_dir))

        return True
        return (num_train_img == _TRAIN_LEN) and \
                (num_train_mask == _TRAIN_LEN) and \
                (num_val_img == _VAL_LEN)

    def _resources(self) -> List[OnlineResource]:
        file_name, sha256 = self._TRAIN_VAL_ARCHIVES['trainval']
        decom_dir = os.path.join(self.root, 'landslide4sense')
        self.decom_dir = decom_dir
        archive = HttpResource("https://syncandshare.lrz.de/dl/fiLurHQ9Cy4NwvmPGYQe7RWM/{}".format(file_name), sha256=sha256)

        if not self.decompress_integrity_check(decom_dir):
            print('Decompressing the tar file...')
            with tarfile.open(os.path.join(self.root, file_name), 'r:gz') as tar:
                tar.extractall(decom_dir)
                tar.close()

        return [archive]

    def _is_in_folder(self, data: Tuple[str, Any], *, name: str, depth: int = 1) -> bool:
        path = pathlib.Path(data)
        in_folder =  name in str(path.parent)
        return in_folder
    
    def _prepare_sample(self, data):
        label_path, label = None, None
        if self.mode=='unsupervised' and self._split=='train':
            image_path, height_path = data
        else:
            (image_path, height_path), label_path = data
        #img = h5py.File(image_path, 'r')['image'][()]
        #img = torch.tensor(img).permute(2, 0, 1)
        #height = h5py.File(height_path, 'r')['image'][()]
        #height = torch.tensor(height)
        #if label_path:
        #    label = h5py.File(label_path, 'r')['image'][()]
        #    label = torch.tensor(label)

        if self._split == 'train' and self.mode == 'unsupervised':
            return (image_path, height_path, None)

        return (image_path, height_path, label_path)

    class _Demux(enum.IntEnum):
        VAL = 0
        TEST = 1
        TRAIN = 2

    def _classify_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        if self._is_in_folder(data, name="train", depth=2):
            return self._Demux.TRAIN
        if self._is_in_folder(data, name="val", depth=2):
            return self._Demux.VAL
        elif self._is_in_folder(data, name="test", depth=2):
            return self._Demux.TEST
        else:
            return None 
    
    def _datapipe(self, res):
        image_dp = FileLister(root=os.path.join(self.root, FNAME, 'images'), recursive=True)
        val_img_dp, test_img_dp, train_img_dp = image_dp.demux(num_instances=3, classifier_fn=self._classify_archive,\
                drop_none=True, buffer_size=INFINITE_BUFFER_SIZE)

        height_dp = FileLister(root=os.path.join(self.root, FNAME, 'heights'), recursive=True)
        val_height_dp, test_height_dp, train_height_dp = height_dp.demux(num_instances=3, classifier_fn=self._classify_archive,\
                drop_none=True, buffer_size=INFINITE_BUFFER_SIZE)

        label_dp = FileLister(root=os.path.join(self.root, FNAME, 'classes'), recursive=True)
        val_label_dp, test_label_dp, train_label_dp = label_dp.demux(num_instances=3, classifier_fn=self._classify_archive,\
                drop_none=True, buffer_size=INFINITE_BUFFER_SIZE)

        train_dp = train_img_dp.zip(train_height_dp).zip(train_label_dp)
        val_dp = val_img_dp.zip(val_height_dp).zip(val_label_dp)
        test_dp = test_img_dp.zip(test_height_dp).zip(test_label_dp)
        
        '''tfs = transforms.Compose(transforms.Resize((256,256)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomVerticalFlip(),
                                 transforms.RandomResizedCrop((224, 224), scale=[0.5, 1]))'''

        ndp = eval(self._split+'_dp')
        ndp = hint_shuffling(ndp)
        ndp = hint_sharding(ndp)
        ndp = Mapper(ndp, self._prepare_sample)
        #ndp = ndp.map(tfs)
        return ndp

    def __len__(self) -> int:
        return {
            'train': _TRAIN_LEN,
            'val': _VAL_LEN,
            'test': _TEST_LEN
        }[self._split]


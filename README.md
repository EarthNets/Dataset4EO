<div  align="center">    
 <img src="resources/datasets4eo.png" width = "400" height = "130" alt="segmentation" align=center />
</div>

# Parse Semantics from Geometry: A Remote Sensing Benchmark for Multi-modal Semantic Segmentation

The proposed benchmark dataset RSMSS can be donwloaded from the following [link](https://syncandshare.lrz.de/getlink/fi8rRALX7JwWtMaSH1jpxiVA/RSUSS.zip):
1. https://syncandshare.lrz.de/getlink/fi8rRALX7JwWtMaSH1jpxiVA/RSUSS.zip
2. https://mediatum.ub.tum.de/1661568
<div  align="center">    
 <img src="resources/RSMSS.png" width = "1000" height = "270" alt="RSMSS" align=center />
</div>



# Contribution Guidelines
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



![example workflow](https://github.com/github/docs/actions/workflows/main.yml/badge.svg)

Todo List

- [x] Re-organize mroe than 180 datasets in Remote sensing cimmunity in a task-oriented way;
- [x] Assign specific datasets to each of the member;
- [X] preprocess the datasets in a AI-ready style;
- [X] re-distribute processed datasets and add citations;
- [X] implementing dataset classes for downloading;
- [ ] implementing the random sampler for geo-spatial datasets;
- [X] supporting for heigh-level repos for specific tasks: obejct detection, segmentation and so forth;
- [X] supporting dataloaders in a easy-to-use way for custom projects;
- [ ] benchmarking all the cutting-edge backbones and task-specific models;

# Supported datasets:

- [x] [RSUSS](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#cityscapes)
- [x] [LandSlide4Sense](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-voc)
- [x] [Eurosat](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k)
- [ ] [XXXX](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context)
- [ ] [XXXX](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context)
- [ ] [XXXX](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context)
- [ ] [XXXX](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context)
- [ ] [XXXX](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context)


## How to use

### Install newest versions of torch and torchdata
```shell
sh install_requirements.sh
```

### Install Dataset4EO
```shell
git clone git@github.com:DeepAI4EO/Dataset4EO.git
python -m pip install -e .
```

### Then it can be used by
```python
from Dataset4EO.datasets import list_datasets, load, landslide4sense
from torch.utils.data import DataLoader2
from tqdm import tqdm

#list all the supported datasets
print(list_datasets())

#create new dataset object by calling:
datasets_dir = './'
dp = landslide4sense.Landslide4Sense(datasets_dir, split='train')

#Then the corresponding dataset will be downloaded and decompressed automatically

#create a dataloader by calling:
data_loader = DataLoader2(dp.shuffle(), batch_size=4, num_workers=4, shuffle=True, drop_last=True)

#Now, iterating the dataloader for training
for it in tqdm(data_loader):
    print(it)
```

# Add Transformations
```python
from Dataset4EO import transforms

tfs = transforms.Compose(transforms.RandomHorizontalFlip(),
                                 transforms.RandomVerticalFlip(),
                                 transforms.RandomResizedCrop((128, 128), scale=[0.5, 1]))
                                 
ndp = ndp.map(tfs)
```

<div  align="center">    
 <img src="resources/datasets4eo.png" width = "400" height = "130" alt="segmentation" align=center />
</div>


![example workflow](https://github.com/github/docs/actions/workflows/main.yml/badge.svg)

Todo List

- [x] Re-organize mroe than 180 datasets in Remote sensing cimmunity in a task-oriented way;
- [x] Assign specific datasets to each of the member;
- [x] preprocess the datasets in a AI-ready style;
- [x] re-distribute processed datasets and add citations;
- [x] implementing dataset classes for downloading;
- [x] supporting for heigh-level repos for specific tasks: obejct detection, segmentation and so forth;
- [x] supporting dataloaders in a easy-to-use way for custom projects;
- [x] benchmarking cutting-edge CV backbones and models on RS data;
- [ ] implementing the random sampler for geo-spatial datasets;

# Supported datasets:

- [x] [DFC2020](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest)
- [x] [LandSlide4Sense](https://www.iarai.ac.at/landslide4sense/)
- [x] [Eurosat](https://github.com/phelber/EuroSAT#)
- [x] [AID](https://captain-whu.github.io/AID/)
- [x] [DIOR](http://www.escience.cn/people/JunweiHan/DIOR.html)
- [x] [DOTA 2.0](https://captain-whu.github.io/DOTA/index.html)
- [x] [fMoW](https://github.com/fMoW/dataset)
- [x] [GeoNRW](https://github.com/gbaier/geonrw)
- [x] [LoveDA](https://github.com/Junjue-Wang/LoveDA)
- [x] [NWPU_VHR10](https://github.com/chaozhong2010/VHR-10_dataset_coco)
- [x] [RSUSS](https://github.com/EarthNets/RSI-MMSegmentation)
- [x] [BigEarthNet](https://bigearth.net/)
- [x] [SEASONET](https://zenodo.org/record/5850307#.Y0cayXbP1D8)
- [x] [SSL4EO_S12](https://github.com/zhu-xlab/SSL4EO-S12)
- [x] [Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)

Continually updating

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


# Contribution Guidelines
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

# Citation
Coming soon

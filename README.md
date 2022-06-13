# Dataset4EO
![example workflow](https://github.com/github/docs/actions/workflows/main.yml/badge.svg)

Todo List

- [x] Re-organize mroe than 180 datasets in Remote sensing cimmunity in a task-oriented way;
- [x] Assign specific datasets to each of the member;
- [ ] preprocess the datasets in a AI-ready style;
- [ ] re-distribute processed datasets and add citations;
- [ ] implementing dataset classes for downloading;
- [ ] implementing the random sampler for geo-spatial datasets;
- [ ] supporting for heigh-level repos for specific tasks: obejct detection, segmentation and so forth;
- [ ] supporting dataloaders in a easy-to-use way for custom projects;
- [ ] benchmarking all the cutting-edge backbones and task-specific models;


## How to use

### Install newest versions of torch and torchdata
```shell
sh install_requirements.sh
python -m pip install -e .
```

### Then it can be used by
```python
from Dataset4EO import datasets
vocdp = datasets.voc.VOC('./')
```

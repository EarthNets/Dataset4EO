![example workflow](https://github.com/github/docs/actions/workflows/main.yml/badge.svg)

Todo List

- [x] Re-organize mroe than 180 datasets in Remote sensing cimmunity in a task-oriented way;
- [x] Assign specific datasets to each of the member;
- [x] preprocess the datasets in a AI-ready style;
- [x] re-distribute processed datasets and add citations;
- [x] implementing dataset classes for downloading;
- [x] implementing the random sampler for geo-spatial datasets;
- [x] supporting for heigh-level repos for specific tasks: obejct detection, segmentation and so forth;
- [x] supporting dataloaders in a easy-to-use way for custom projects;
- [x] benchmarking all the cutting-edge backbones and task-specific models;


# Dataset4EO
Re-organize the datasets of remote sensing datasets 

#Install newest versions of torch and torchdata
```shell
sh install_requirements.sh
python -m pip install -e .
```

#Then it can be used by
```python
from Dataset4EO import datasets
vocdp = datasets.voc.VOC('./')
```

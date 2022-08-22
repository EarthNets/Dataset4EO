pip uninstall -y torch torchvision torchdata
pip install -y --pre torch torchvision torchdata --extra-index-url https://download.pytorch.org/whl/nightly/cu113
pip install -y mmcv-full==1.6.0
pip install prettytable
pip install pycocotools
pip install wandb

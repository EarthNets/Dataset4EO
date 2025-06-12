import dataset4eo as eodata
import litdata as ld #type: ignore
import time


train_dataset = eodata.StreamingDataset('/datastore01/DATA/3D/PLANET/sining_code/exebench_fire/train',num_channels=6, channels_to_select=[0, 1, 2], shuffle=True, drop_last=True)
train_dataloader = ld.StreamingDataLoader(train_dataset)

iters = 0
start = time.time()
for sample in train_dataloader:
    base_name, img, cls, height = sample["name"], sample['image'], sample['class'], sample["spatial_coords"]
    print("image", img.shape)
    print("class", cls.shape)
    print("spatial_coords", height.shape)
    iters += 1
    if iters == 100:
        break
    
end = time.time()
print(end-start)

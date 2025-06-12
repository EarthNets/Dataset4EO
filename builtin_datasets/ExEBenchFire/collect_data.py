import os
import rasterio
from PIL import Image
from torch.utils.data import Dataset
import json
import numpy as np
import dataset4eo as eodata

class DataClass(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        super(DataClass, self).__init__()

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.base_dir = os.path.join(root_dir, split)

        self.image_files = sorted([f for f in os.listdir(self.base_dir) if f.endswith('merged.tif')])
        self.bands = [0, 1, 2, 3, 4, 5]
        self.means = np.array([0.0333, 0.0570, 0.0589, 0.2323, 0.1973, 0.1194])
        self.stds = np.array([0.0227, 0.0268, 0.0400, 0.0779, 0.0871, 0.0724])

    def __len__(self):
        return len(self.image_files)

    def normalize(self, image):
        return (image - self.means[:, None, None]) / self.stds[:, None, None]

    def __getitem__(self, idx):
        img_file = os.path.join(self.base_dir, self.image_files[idx])
        basename = os.path.basename(img_file).split("_merged.tif")[0]
        label_file = img_file.replace("_merged.tif", ".mask.tif")

        with rasterio.open(img_file) as dataset:
            image = dataset.read()
            image = np.stack([image[b] for b in self.bands], axis=0)
            image = self.normalize(image)
            image = image.transpose(1, 2, 0).astype(np.float16)
            try:
                spatial_coords = np.array(dataset.lnglat())
            except Exception:
                spatial_coords = None

        label = np.array(Image.open(label_file))
        label[label == -1] = 0
        label = label[np.newaxis, :, :]

        print(image.shape, label.shape)

        return {
            "name": basename,
            "image": image,
            "class": label,
            "spatial_coords": spatial_coords
        }

###########################-Create optimized Dataset-###############################
def process_sample_fn(dataset, index):
    sample = dataset[index]
    image = sample["image"]
    class_data = sample["class"].astype(np.uint8)
    return {
        "name": sample["name"],
        "image": image,
        "class": class_data,
        "spatial_coords": sample["spatial_coords"].astype(np.float32)
    }

class ProcessSampleWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, index):
        return process_sample_fn(self.dataset, index)

if __name__ == "__main__":
    root_dir = '/datastore01/DATA/3D/PLANET/sining_code/ExEBench/fire'
    output_folder = f'/datastore01/DATA/3D/PLANET/sining_code/exebench_fire'

    for split in ["train", "test"]:
        output_dir = os.path.join(output_folder, split)
        os.makedirs(output_dir, exist_ok=True)

        output_flag = os.path.join(os.path.dirname(os.path.abspath(__file__)), split+".finish")
        if os.path.exists(output_flag):
            continue

        dataset = DataClass(root_dir=root_dir, split=split)

        metadata = {
            "Dataset": "ExEBench.Fire",
            "split": split,
            "num_samples": len(dataset),
            "attributes": {
                "name": {"dtype": "str"},
                "image": {"dtype": "float32", "format": "numpy"},
                "class": {"dtype": "uint8", "format": "numpy"},
                "spatial_coords": {"dtype": "float32", "format": "numpy"},
                "classes": {
                    0: "Not burned",
                    1: "Burn scar"
                }
            }
        }
        process_fn = ProcessSampleWrapper(dataset)

        eodata.optimize(
            fn=process_fn,
            inputs=list(range(len(dataset))),
            output_dir=output_dir,
            num_workers=8,
            chunk_bytes="256MB",
        )

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        print(f"{split.capitalize()} dataset optimized and metadata saved to {metadata_path}")

        with open(output_flag, "w") as f:
            f.writelines("finished! \n")


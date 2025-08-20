
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def load_data_from_folder(label_file):
    labels = np.load(label_file, allow_pickle=True)
    return labels


class STdata(Dataset):
    def __init__(self, label, root=None, transform=None, is_raw=False):
        self.transform = transform
        self.root = root
        self.label = label
        self.is_raw = is_raw

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = self.label[idx]['label']
        if self.is_raw:
            img_path = self.label[idx]['img_path']
            label = np.expand_dims(label, axis=0)
            image = Image.open(img_path).convert("RGB").resize((112, 112))
        else:
            img_path = '.'+self.label[idx]['img_path']
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, np.log1p(label.astype(np.float32))


class ImageGraphDataset(Dataset):
    def __init__(self, data_infor_path, transform=None):
        self.data_list = np.load(data_infor_path, allow_pickle=True).tolist()
        self.transform = transform
        self.name = data_infor_path.split('/')[-2]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path = '.'+self.data_list[idx]['img_path']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.data_list[idx]['label']
        return image, label.astype(np.float32)


def create_dataloaders_for_each_file(npy_file_paths, batch_size=256, transform=None):
    dataloaders = {}
    for npy_file in npy_file_paths:
        dataset = ImageGraphDataset(npy_file, transform=transform)
        dataloaders[npy_file] = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloaders
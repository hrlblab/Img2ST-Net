import os
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
            raw_path = self.label[idx]['img_path']
            if '../datasets/' in raw_path:
                rel_part = raw_path.split('../datasets/')[-1]
                img_path = os.path.join(os.path.dirname(self.root), rel_part)
            else:
                img_path = raw_path
            label = np.expand_dims(label, axis=0)
            image = Image.open(img_path).convert("RGB").resize((224, 224))
        else:
            img_path = '.'+self.label[idx]['img_path']
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, np.log1p(label.astype(np.float32))


class STDataset(Dataset):
    def __init__(self, data, transform=None, root_path=None):
        self.data = data
        self.transform = transform
        self.root_path = root_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['img_path']
        if self.root_path:
            parts = img_path.split('cropped_img/')
            if len(parts) > 1:
                img_path = os.path.join(self.root_path, 'cropped_img', parts[-1])

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = np.log1p(item['label'].astype(np.float32))
        return image, label


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





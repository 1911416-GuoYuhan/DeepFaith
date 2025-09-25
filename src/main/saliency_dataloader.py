import os
import numpy as np
import torch
from generate_faithfulness import ts
from generate_saliency import get_dataset_name
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SaliencyDataset(Dataset):
    def __init__(self, image_dir, saliency_dir, img_ext='.jpg', saliency_ext='.npy'):
        self.image_dir = image_dir
        self.saliency_dir = saliency_dir
        self.img_ext = img_ext
        self.saliency_ext = saliency_ext
        
        self.filenames = [
            os.path.splitext(f)[0] 
            for f in os.listdir(image_dir) 
            if f.endswith(img_ext)
        ]
        dataset_name = get_dataset_name(image_dir)
        self.transform = ts[dataset_name]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        base_name = self.filenames[idx]
        
        img_path = os.path.join(self.image_dir, base_name + self.img_ext)
        image = Image.open(img_path).convert('RGB')
        
        saliency_path = os.path.join(self.saliency_dir, base_name + self.saliency_ext)
        saliency = np.load(saliency_path).astype(np.float32)
    
        return {
            'x': self.transform(image),
            'a': torch.from_numpy(saliency)
        }

class TextSaliencyDataset(Dataset):
    def __init__(self, text_dir, saliency_dir, text_ext='.npy', saliency_ext='.npy', transform=None):
        self.text_dir = text_dir
        self.saliency_dir = saliency_dir
        self.text_ext = text_ext
        self.saliency_ext = saliency_ext
        self.transform = transform

        self.filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(text_dir)
            if f.endswith(text_ext)
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        base_name = self.filenames[idx]

        text_path = os.path.join(self.text_dir, base_name + self.text_ext)
        text = np.load(text_path,allow_pickle=True).astype(np.int64)  

        saliency_path = os.path.join(self.saliency_dir, base_name + self.saliency_ext)
        saliency = np.load(saliency_path).astype(np.float32)

        if self.transform:
            text = self.transform(text)

        return {
            'x': torch.from_numpy(text),
            'a': torch.from_numpy(saliency)
        }


if __name__ == '__main__':
    dataset = SaliencyDataset(
        image_dir='../../data/saliency_dateset/oct/resnet/sample',
        saliency_dir='../../data/saliency_dateset/oct/resnet/saliency'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    
    for batch in dataloader:
        x_batch = batch['x']
        a_batch = batch['a']
        
        print(x_batch,a_batch )
        print(x_batch.shape, a_batch.shape)
        break
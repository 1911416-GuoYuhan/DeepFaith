from typing import List, Optional, Tuple
import os
import cv2
from PIL import Image
import hydra
import numpy as np
import pyrootutils
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from tqdm.auto import tqdm
from main_explain import _compute_explain_data_for_model, _set_random_seed
from torchvision.transforms.functional import to_pil_image
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from torchvision.transforms import Normalize
from torchvision.utils import save_image
from src.modules.models import ModelsModule
from src.modules.xai_methods import XAIMethodsModule
from src import utils

log = utils.get_pylogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_filenames_from_dataloader(dataloader, dataset_type='imagenet'):

    if hasattr(dataloader.dataset, 'samples'):

        return [os.path.basename(path) for path, _ in dataloader.dataset.samples]
    
    elif hasattr(dataloader.dataset, 'image_paths'):

        return [os.path.basename(path) for path in dataloader.dataset.image_paths]
    
    else:

        return [f"sample_{i:06d}.jpg" for i in range(len(dataloader.dataset))]

class DataLoaderWrapper:

    
    def __init__(self, original_dataloader, filenames: List[str] = None):
        self.original_dataloader = original_dataloader
        self.filenames = filenames or self._generate_filenames()
        self.batch_size = original_dataloader.batch_size
    
    def _generate_filenames(self):

        dataset_size = len(self.original_dataloader.dataset)
        return [f"sample_{i:06d}.jpg" for i in range(dataset_size)]
    
    def __iter__(self):
        filename_idx = 0
        for batch_data in self.original_dataloader:
            x_batch, y_batch = batch_data
            batch_size = x_batch.size(0)
            
            # 获取当前批次的文件名
            batch_filenames = self.filenames[filename_idx:filename_idx + batch_size]
            filename_idx += batch_size
            
            yield x_batch, y_batch, batch_filenames
    
    def __len__(self):
        return len(self.original_dataloader)


IMAGENET_DENORMALIZE = Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def get_dataset_name(str):
    if 'imagenet' in str:
        return 'imagenet'
    if 'oct' in str:
        return 'oct'
    if 'resisc' in str:
        return 'resisc'
    
def get_model_name_by_idx(idx):
    models = {
        0:'resnet',
        1:'efficientnet',
        2:'deit'
    }
    return models[idx]

def save_saliency_map(saliency_map,file_path,np_path):

    if len(saliency_map.shape) == 3:
        if saliency_map.shape[0] in [1, 3]:  # (C, H, W)
            map_data = np.mean(saliency_map, axis=0)  
        else:  # (H, W, C)
            map_data = np.mean(saliency_map, axis=2)  

    # 标准化到 [0, 255]
    map_data = map_data - map_data.min()
    if map_data.max() > 0:
        map_data = map_data / map_data.max()
    map_data = (map_data * 255).astype(np.uint8)

    # print(saliency_map.shape,saliency_map.max(),saliency_map.min())

    colored_map = cv2.applyColorMap(map_data, cv2.COLORMAP_JET)
    colored_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)

   
    Image.fromarray(colored_map).save(file_path)
    np.save(np_path,saliency_map)

saliency_dir = 'generated_saliency'
sub_dir_saliency = 'saliency'
sub_dir_numpy = 'numpy'

@utils.task_wrapper
def generate_saliency(cfg: DictConfig):
    def denormalize(tensor):
     
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean
    
    """Main explanation loop."""
    # Set seed for random number generators
    _set_random_seed(cfg)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    original_dataloader = datamodule.dataloader()
    models = ModelsModule(cfg)
    data_dir = cfg.paths.data_dir
    dataset_name = get_dataset_name(cfg.data._target_)

    filenames = get_filenames_from_dataloader(original_dataloader, dataset_type='imagenet')
    # 包装数据加载器以包含文件名
    dataloader = DataLoaderWrapper(original_dataloader, filenames)
    

    for batch_idx, (x_batch, y_batch, batch_filenames) in enumerate(tqdm(dataloader, desc="batches")):
  
        # Load pretrained models
        explain_data = []
        # Loop over models to compute saliency maps
        log.info(f"Starting saliency map computation for each Model and XAI Method")
        for model in tqdm(
            models.models, desc=f"Attribution for {cfg.data.modality} Models", colour="BLUE"
        ):
            print(type(model))
            model.eval()
            explain_data_model = _compute_explain_data_for_model(
                cfg, model, x_batch, y_batch
            )
            explain_data.append(np.vstack(explain_data_model))
        # [3*(10, 14, 3, 224, 224)]
        for idx in range(len(explain_data)):
            model_saliency = explain_data[idx]
            model_name = get_model_name_by_idx(idx)
            # (10, 14, 3, 224, 224)
            for sample_idx, sample_saliency in enumerate(model_saliency):
                sample_name = batch_filenames[sample_idx]
                # 14, 3, 224, 224
                output_dir = os.path.join(data_dir,saliency_dir,dataset_name,model_name,sample_name)
                #print(output_dir)
                os.makedirs(output_dir, exist_ok=True)
                Xai_methods = list(cfg.explain_method.keys())
                for saliency_idx ,saliency_map in enumerate(sample_saliency):
                    # 3, 224, 224
                    filename = Xai_methods[saliency_idx]+'.jpg'
                    saliency_name = Xai_methods[saliency_idx]
                    filepath = os.path.join(output_dir,sub_dir_saliency,filename)
                    os.makedirs(os.path.join(output_dir,sub_dir_saliency), exist_ok=True)
                    np_path = os.path.join(output_dir,sub_dir_numpy, saliency_name)
                    os.makedirs(os.path.join(output_dir,sub_dir_numpy), exist_ok=True)
                    save_saliency_map(saliency_map,filepath,np_path)
                
                filename = 'original.jpg'
                original_image = x_batch[sample_idx]
                if dataset_name == 'imagenet' or dataset_name == 'resisc':
                    original_image = denormalize(original_image).clamp(0, 1)
                save_image(original_image, os.path.join(output_dir,filename))
        break

@hydra.main(
    version_base="1.3", config_path=os.getcwd() + "/configs", config_name="explain.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    generate_saliency(cfg)

if __name__ == "__main__":
    main()
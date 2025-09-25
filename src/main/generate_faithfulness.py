import os
import pyrootutils
import hydra
import torch
import numpy as np
import json
import gc 
import logging
from typing import List, Optional, Tuple
from omegaconf import DictConfig
from torchvision import transforms
from PIL import Image
from main_eval_GPU import _prepare_tensor, _evaluate_single_model
from src.modules.models import ModelsModule
from generate_saliency import get_dataset_name, get_model_name_by_idx

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src import utils


ts = {
    'imagenet': transforms.Compose([
        transforms.ToTensor(),  
        transforms.Resize((224, 224)),  
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ]),
    'oct': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
    ]),
    'resisc': transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),  
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])
}


original_img_dir = '../../data/generated_saliency'
original_img_name = 'original.jpg'


saliency_sub_dir = 'numpy'
dict_name = 'faithfulness.json'

def generate_saliency_on_single_model(
    model,model_idx,dataset_name,Xai_methods,device,batch_size,cfg,eval_metrics
    ):
    model_name = get_model_name_by_idx(model_idx)
    sample_dir = os.path.join(original_img_dir,dataset_name,model_name)

    xai_used = Xai_methods if model_name == 'deit' else Xai_methods[:14]
    #print(len(xai_used))

    model = model.to(device)
    sample_num = len(os.listdir(sample_dir))

    for idx in range(0,sample_num,batch_size):
        dirs = os.listdir(sample_dir)[idx:idx+batch_size]

        image_tensor_list = []
        saliency_numpy_list = []

        for dir in dirs:
            image_path = os.path.join(sample_dir,dir,original_img_name)
            image = Image.open(image_path)
            image_tensor = ts[dataset_name](image)
            # torch.Size([3, 224, 224])
            # print(image_tensor.shape)
            image_tensor_list.append(image_tensor)

            np_dir = os.path.join(sample_dir,dir,saliency_sub_dir)
            saliency_maps = os.listdir(np_dir)
            saliency_np_list_sample = []
            for saliency_path in saliency_maps:
                saliency_np = np.load(os.path.join(np_dir,saliency_path))
                saliency_np_list_sample.append(saliency_np)
            # (Xai_methods,3,224,224)
            saliency_numpy_list.append(np.stack(saliency_np_list_sample))
            
            image.close()
            del image, saliency_np_list_sample

            # logging.info('Sample '+dir+' eval ready.')
            # print('Sample',dir,'eval ready.')
        
        explain_data_single_model = torch.from_numpy(np.stack(saliency_numpy_list)).to(device)
        del saliency_numpy_list
        explain_data = [explain_data_single_model] * 3

        x_batch = _prepare_tensor(torch.stack(image_tensor_list),cfg)
        del image_tensor_list
        

        with torch.no_grad():
            y_batch = torch.argmax(model(x_batch),dim=1)

        
        # print(explain_data[0].shape,x_batch.shape,y_batch.shape,y_batch)
        explanation_single_batch = _evaluate_single_model(cfg, model, model_idx, explain_data, x_batch, y_batch)
        
        logging.info('Batch eval done.')
        
        for idx_dir, dir in enumerate(dirs):
            json_save_path = os.path.join(sample_dir,dir,dict_name)
            log_dict = {}

            for idx_xai, xai in enumerate(xai_used):
                log_dict[xai] = {}
                for idx_eval, evaluation in enumerate(eval_metrics):
                    explanation_single_xai = explanation_single_batch[idx_xai].reshape(len(eval_metrics),batch_size)
                    # print(idx_xai,idx_eval,idx_dir)
                    # print(explanation_single_xai.shape)
                    # print(explanation_single_xai.shape,explanation_single_xai[idx_eval])
                    log_dict[xai][evaluation] = explanation_single_xai[idx_eval][idx_dir].item()

            with open(json_save_path, 'w') as f:
                json.dump(log_dict, f, indent=2)
            # print('Json saved at',json_save_path)
        
        del x_batch, y_batch, explain_data, explanation_single_batch
        
        gc.collect()
        logging.info('Batch samples '+str(len(dirs))+' eval saved.')
        # print('Batch samples',dirs,'eval saved.')
    
    del model
        
    gc.collect()
    logging.info(f'Model {model_name} completed, memory cleaned')
    # print(f'Model {model_name} completed, memory cleaned')

@utils.task_wrapper
def generate_saliency_for_models(cfg: DictConfig):
    # print(cfg)

    batch_size = cfg.data.batch_size
    dataset_name = get_dataset_name(cfg.data._target_)

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        filename=os.path.join('/home/gyh/ryy/code/latec/logs',dataset_name,'denoise.log'), 
        filemode='a' 
    )
    models = ModelsModule(cfg)
    Xai_methods = list(cfg.explain_method.keys())
    eval_metrics = list(cfg.eval_metric.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_idx, model in enumerate(models.models):
        generate_saliency_on_single_model(
            model,model_idx,dataset_name,Xai_methods,device,batch_size,cfg,eval_metrics
            )
    
    del models

    gc.collect()
    print('All processing completed, final memory cleanup done')

@utils.task_wrapper
def generate_saliency_for_singel_model(cfg: DictConfig):
    model_idx = 2
    batch_size = cfg.data.batch_size
    dataset_name = get_dataset_name(cfg.data._target_)

    logging.basicConfig(
        level=logging.INFO,  
        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
        filename=os.path.join('/home/gyh/ryy/code/latec/logs',dataset_name,'denoise.log'),  
        filemode='a' 
    )
    models = ModelsModule(cfg)
    Xai_methods = list(cfg.explain_method.keys())
    eval_metrics = list(cfg.eval_metric.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.models[model_idx]

    generate_saliency_on_single_model(
        model,model_idx,dataset_name,Xai_methods,device,batch_size,cfg,eval_metrics
        )

    del models

    gc.collect()
    print('All processing completed, final memory cleanup done')


@hydra.main(
    version_base="1.3", config_path=os.getcwd() + "/configs", config_name="eval.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    # generate_saliency_for_models(cfg)
    # 0: resnet 1: efficientnet 2: deit
    generate_saliency_for_singel_model(cfg)

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"
    main()
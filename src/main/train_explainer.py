import os
import torch
import pyrootutils
import hydra
import numpy as np
import math
import random
import json
import time

from datetime import datetime
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from torch.nn import MSELoss
from typing import List, Optional, Tuple
from omegaconf import DictConfig
from generate_faithfulness import ts
from saliency_dataloader import SaliencyDataset
from loss_func import SuperviseLoss, FaithfulLoss

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.modules.components.resnet import resnet50
from src.modules.components.deit_vit import deit_small_patch16_224
from src.main.image_patch import PatchEmbed, PatchRecover
from src.explainer.model.transformer import TransformerEncoder
from src.modules.eval_metrics import EvalMetricsModule
from src.modules.models import load_from_lightning


class CVwarper(nn.Module):
    def __init__(self, model,embeder,recover):
        super(CVwarper, self).__init__()
        self.model = model
        self.embeder = embeder
        self.recover = recover

    def forward(self, x):
        return self.model(self.recover(x))
        

def get_eval_batch(data_dir, batch_size, dataset_name, device):
    
    all_samples = [s for s in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, s))]
    
    selected_samples = random.sample(all_samples, batch_size)
    x_list = []
    faithfulness_list = []
    transform = ts[dataset_name]
    for sample in selected_samples:
        sample_dir = os.path.join(data_dir, sample)
        img_path = os.path.join(sample_dir, 'original.jpg')
        json_path = os.path.join(sample_dir, 'faithfulness.json')
  
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)  # [C,H,W]
        x_list.append(img_tensor)

        with open(json_path, 'r') as f:
            faithfulness = json.load(f)
        faithfulness_list.append(faithfulness)
    # 拼成batch
    x_batch = torch.stack(x_list, dim=0).to(device)
    return x_batch, faithfulness_list

def load_model_to_explain(model_name,cfg):
    def _create_resnet50(cfg):
        model = resnet50()
        model.fc = nn.Linear(2048, cfg.data.num_classes, bias=True)
        load_from_lightning(model, cfg.data.weights_resnet)
        return model

    def _create_efficientnet_b0(cfg):
        model = efficientnet_b0()
        model.classifier[1] = nn.Linear(1280, cfg.data.num_classes, bias=True)
        load_from_lightning(model, cfg.data.weights_effnet)
        return model

    def _create_vit(cfg):
        model = deit_small_patch16_224(num_classes=cfg.data.num_classes)
        load_from_lightning(model, cfg.data.weights_vit)
        return model
        
    if isinstance(cfg.data.weights_vit, bool):
        if model_name == 'resnet':
            return resnet50(weights=cfg.data.weights_resnet)
        if model_name == 'efficientnet':
            return efficientnet_b0(weights=cfg.data.weights_effnet)
        if model_name == 'deit':
            return deit_small_patch16_224(pretrained=cfg.data.weights_vit)

    else:  # For all other Image datasets
        if model_name == 'resnet':
            return _create_resnet50(cfg)
        if model_name == 'efficientnet':
            return _create_efficientnet_b0(cfg)
        if model_name == 'deit':
            return _create_vit(cfg)


def eval_explainer(explainer,dataset_name,model_name,device,patch_embed,patch_recover,cfg,model,log_file):
    explainer.eval()
    eval_data_dir = f'../../data/explanation_dataset/{dataset_name}/{model_name}/test'
    x_batch, faithfulness_list = get_eval_batch(
        eval_data_dir,
        batch_size=TrainConfig.batch_size,
        dataset_name=dataset_name,
        device=device
    )
    inputs = patch_embed(x_batch)
    outputs = explainer(inputs)
    a_batch = patch_recover(outputs)
    # a_batch = outputs.unsqueeze(-1).repeat(1,1,768)
    # x_batch = inputs.mean(dim=-1)
    model = CVwarper(model,patch_embed,patch_recover).to(device)
    model.eval()

    eval_methods = EvalMetricsModule(cfg, model)
    y_batch = torch.argmax(model(inputs),dim=1)

    scores = eval_methods.evaluate(
        model,
        inputs,
        y_batch,
        a_batch,
        # xai_methods,
        # idx_xai,
        custom_batch=[
            x_batch,
            y_batch,
            a_batch,
            list(range(TrainConfig.batch_size)),
        ],
    )

    avg_score=[]
    for score in scores:
        avg_score.append(round(torch.mean(score).item(),3))
    print('DeepFaith',avg_score)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write('DeepFaith '+str(avg_score)+'\n')

    compare_scores={}
    for faithfulness_dict in faithfulness_list:
        for xai_method in faithfulness_dict.keys():
            if xai_method not in compare_scores.keys():
                compare_scores[xai_method] = {}

            faithfulness_scores = faithfulness_dict[xai_method]
            for metric, faithfulness_score in faithfulness_scores.items():
                if metric not in compare_scores[xai_method].keys():
                    compare_scores[xai_method][metric] = []
                compare_scores[xai_method][metric].append(faithfulness_score)

    all_method_scores = {}
    all_method_scores['DeepFaith'] = avg_score
    
    for xai_method in compare_scores.keys():
        avg_compare_scores = []
        for metric in compare_scores[xai_method].keys():
            avg_compare_scores.append(round(np.mean(compare_scores[xai_method][metric]),3))
        print(xai_method,avg_compare_scores)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(str(xai_method)+' '+str(avg_compare_scores)+'\n')
        all_method_scores[xai_method] = avg_compare_scores

    if all_method_scores:

        method_names = list(all_method_scores.keys())
        num_columns = len(next(iter(all_method_scores.values())))
        
        if all_method_scores:
            method_names = list(all_method_scores.keys())
            num_columns = len(next(iter(all_method_scores.values())))
           
            rankings = {method: [] for method in method_names}
            for col_idx in range(num_columns):
                col_scores = {method: scores[col_idx] for method, scores in all_method_scores.items()}
                is_minimize_column = (col_idx == num_columns - 3) or (col_idx == num_columns - 5)
                if is_minimize_column:
                    sorted_methods = sorted(col_scores.items(), key=lambda x: x[1])
                else:
                    sorted_methods = sorted(col_scores.items(), key=lambda x: x[1], reverse=True)
              
                for rank, (method, score) in enumerate(sorted_methods, 1):
                    rankings[method].append(rank)

            avg_rankings = {method: round(np.mean(ranks), 3) for method, ranks in rankings.items()}
   
            table = []
            header = ['Method'] + [f'{i+1}rank' for i in range(num_columns)] + ['avg_rank']
            table.append(header)
            for method in method_names:
                row = [method] + [str(r) for r in rankings[method]] + [str(avg_rankings[method])]
                table.append(row)
            col_widths = [max(len(str(row[i])) for row in table) for i in range(len(header))]
  
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write('\n=== rank_all_methods ===\n')
                for row in table:
                    line = '  '.join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
                    f.write(line + '\n')
                f.write('==================\n\n')

    # print(scores)
    explainer.train()

def train_explainer(datadir,dataset_name,model_name,cfg):
 
    image_dir = os.path.join(datadir,dataset_name,model_name,'train','sample')
    saliency_dir = os.path.join(datadir,dataset_name,model_name,'train','saliency')

    dataset = SaliencyDataset(
        image_dir=image_dir,
        saliency_dir=saliency_dir
    )
    dataloader = DataLoader(
        dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=True,
        num_workers=TrainConfig.num_workers
    )

    device = TrainConfig.device

    explainer = TransformerEncoder(
        d_model=model_config['patch_embed_dim'], 
        n_head=model_config['n_head'], 
        seq_len=model_config['hw_dim'], 
        max_len=model_config['max_len'],
        ffn_hidden=model_config['ffn_hidden'], 
        n_layers=model_config['n_layers'], 
        drop_prob=model_config['drop_prob'], 
        device=device
    ).to(device)

    supervise_loss = SuperviseLoss(corr=TrainConfig.corr).to(device)
    faithful_loss = FaithfulLoss(
        nr_runs=TrainConfig.nr_runs, corr=TrainConfig.corr, 
        perturb_effect = TrainConfig.perturb_effect, 
        perturbation = TrainConfig.perturbation).to(device)

    patch_embed = PatchEmbed(image_size=TrainConfig.image_size, 
    patch_size=TrainConfig.patch_size, in_c=TrainConfig.in_c, 
    embed_dim=TrainConfig.embed_dim).to(device)
    patch_recover = PatchRecover(original_size=TrainConfig.image_size, 
    patch_size=TrainConfig.patch_size, out_c=TrainConfig.in_c, 
    embed_dim=TrainConfig.embed_dim).to(device)

    optimizer = torch.optim.Adam(
        explainer.parameters(),
        lr=TrainConfig.lr,
        weight_decay=TrainConfig.weight_decay
    )
    cfg.data.batch_size = TrainConfig.batch_size
    model = load_model_to_explain(model_name,cfg).to(device)
    model.eval()

    model = CVwarper(model,patch_embed,patch_recover).to(device)
    model.eval()

    def loss_weight(t,t0):
        axis = t-t0
        return 1-1/(1+math.exp(-axis/TrainConfig.divider))


    loss_every_epoch = []
    Lpc_every_epoch = []
    Llc_every_epoch = []
    alpha_every_epoch = []

    weight_flag = False
    faith_finish = False
    begin_epoch = 0

    log_dir = '../../logs'
    model_save_dir = '../../explainers'
    
    now = datetime.now()
    
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(log_dir,dataset_name,model_name,time_str+'.txt')

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('Time:'+str(time.time())+'\n')
        f.write('model_config: '+json.dumps(model_config, ensure_ascii=False)+'\n')
        train_config_dict = {k: v for k, v in TrainConfig.__dict__.items() if not k.startswith('__') and not callable(getattr(TrainConfig, k))}
        f.write('TrainConfig: '+json.dumps(train_config_dict, ensure_ascii=False)+'\n')

    for epoch in range(TrainConfig.epochs):
        print('Alpha',TrainConfig.alpha)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('Alpha '+str(TrainConfig.alpha)+'\n')

        loss_scores = []
        Lpcs = []
        Llcs = []
        alpha_every_epoch.append(TrainConfig.alpha)

        for batch in dataloader:
            x = batch['x'].to(device)
            a = batch['a'].to(device)
            # print(x.shape)
            inputs = patch_embed(x)
            outputs = explainer(inputs)

            s = outputs # [batch_size, seq_len]

            # print(inputs)
            a_min = a.amin(dim=(1,2,3), keepdim=True)
            a_max = a.amax(dim=(1,2,3), keepdim=True)
            a = (a - a_min) / (a_max - a_min + 1e-8)
            a = patch_embed(a)
            
            if TrainConfig.alpha == 1:
                total_loss = supervise_loss(s,a)
                Lpcs.append(total_loss.item())
                Llcs.append(0)
            elif TrainConfig.alpha == 0:
                total_loss = faithful_loss(inputs, model, s)
                Llcs.append(total_loss.item())
                Lpcs.append(0)
            else:
                loss_s = supervise_loss(s, a)
                loss_f = faithful_loss(inputs, model, s)
                total_loss = TrainConfig.alpha * loss_s + (1-TrainConfig.alpha) * loss_f
                Lpcs.append(loss_s.item())
                Llcs.append(loss_f.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
            loss_scores.append(total_loss.item())

        mean_loss = np.mean(loss_scores)
        print(f'Epoch {epoch+1}/{TrainConfig.epochs} Loss: {mean_loss:.4f}')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f'Epoch {epoch+1}/{TrainConfig.epochs} Loss: {mean_loss:.4f}\n')
        loss_every_epoch.append(mean_loss)
        Lpc_every_epoch.append(np.mean(Lpcs))
        Llc_every_epoch.append(np.mean(Llcs))

        if len(Lpc_every_epoch) >= 5:
            last5 = Lpc_every_epoch[-5:]
            var_last5 = np.var(last5)
            if not weight_flag:
                if var_last5 <= TrainConfig.eps:
                    weight_flag = True
                    begin_epoch = epoch
                    print(' Faithfulness loss convergence')
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(' Faithfulness loss convergence\n')
            if var_last5 > TrainConfig.divider*TrainConfig.eps:
                weight_flag = False
                TrainConfig.alpha = 1
        if weight_flag:
            TrainConfig.alpha = loss_weight(epoch,begin_epoch)

        torch.save(explainer.state_dict(), 
        os.path.join(model_save_dir,dataset_name,model_name,'epoch_'+str(epoch+1)+'_explainer_model.pth'))

        if (epoch+1) % 5 == 0 and epoch != 1:
            print('Begin Faitfulness Evaluation')
            eval_explainer(explainer,dataset_name,model_name,device,patch_embed,patch_recover,cfg,model,log_file)



model_config = {
    'hw_dim': 196,      # Number of patches (14*14 for ViT-B/16, 224x224)
    'patch_embed_dim': 768, # Patch embedding dimension for ViT-Base (this is P_embed_dim for Transformer d_model)
    'n_head': 8,        # Transformer heads
    'max_len': 1000,     # Max sequence length for PositionalEncoding, should be >= hw_dim
    'ffn_hidden': 1024, # Transformer FFN hidden size
    'n_layers': 6,      # Number of Transformer encoder layers
    'drop_prob': 0.5,   # Dropout probability
}

class TrainConfig:
    batch_size = 32
    epochs = 1000
    alpha = 1
    image_size = 224 
    patch_size = 16 
    in_c = 3 
    embed_dim = 768
    num_workers=4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # training parameter
    lr = 1e-5
    weight_decay = 1e-5
    nr_runs=10
    corr='pearson'
    perturb_effect = 'minus'
    perturbation = 'mask'
    eps = 1e-5
    stop = 0.4
    divider = 10


@hydra.main(
    version_base="1.3", config_path=os.getcwd() + "/configs", config_name="eval.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    datadir='../../data/explanation_dataset'
    dataset_name='imagenet'
    model_name='resnet'

    train_explainer(datadir,dataset_name,model_name,cfg)

def set_seed(seed):
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False   
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    set_seed(42)
    main()


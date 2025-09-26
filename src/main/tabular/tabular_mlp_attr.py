import torch
from captum.attr import (
    IntegratedGradients, GradientShap, DeepLift, Occlusion, FeatureAblation, Saliency, Lime, KernelShap
)
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from train_mlp_model import pre_step,model_setting, MLP

# ==== Tabular Dataset ====
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return {'features': self.X[idx], 'label': self.y[idx]}

# ==== Captum Explainer ====
def get_explainer(method, model):
    if method == 'IntegratedGradients':
        return IntegratedGradients(model)
    elif method == 'GradientShap':
        return GradientShap(model)
    elif method == 'DeepLift':
        return DeepLift(model)
    elif method == 'Occlusion':
        return Occlusion(model)
    elif method == 'FeatureAblation':
        return FeatureAblation(model)
    elif method == 'Saliency':
        return Saliency(model)
    elif method == 'Lime':
        return Lime(model)
    elif method == 'KernelShap':
        return KernelShap(model)
    else:
        raise ValueError(f"Unknown explanation method: {method}")


def explain_batch(data_loader, model, method='Saliency', device='cpu', max_batches=None, batch_size=32):
    model.eval()
    model.to(device)
    explainer = get_explainer(method, model)
    all_attributions = []
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Explanation progress")):
        features = batch['features'].to(device)
        labels = model(features).argmax(dim=1)
        if method in ['IntegratedGradients', 'GradientShap', 'DeepLift', 'Saliency']:
            if method == 'IntegratedGradients':
                baseline = torch.zeros_like(features).to(device)
                attributions = explainer.attribute(features, baseline, target=labels, n_steps=20)
            elif method == 'GradientShap':
                baseline = torch.zeros_like(features).to(device)
                attributions = explainer.attribute(features, baselines=baseline, target=labels)
            elif method == 'DeepLift':
                baseline = torch.zeros_like(features).to(device)
                attributions = explainer.attribute(features, baselines=baseline, target=labels)
            elif method == 'Saliency':
                attributions = explainer.attribute(features, target=labels)
        elif method == 'Occlusion':
            attributions = explainer.attribute(features, sliding_window_shapes=(1,), baselines=0, target=labels)
        elif method == 'FeatureAblation':
            attributions = explainer.attribute(features, baselines=0, target=labels)
        elif method == 'Lime':
            attributions = explainer.attribute(features, target=labels, n_samples=20)
        elif method == 'KernelShap':
            attributions = explainer.attribute(features, target=labels, n_samples=20)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
        all_attributions.append(attributions.detach().cpu())
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    all_attributions = torch.cat(all_attributions, dim=0)
    return all_attributions.numpy()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except ImportError:
        pass

    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    csv_path = "../../../data/datasets/tabular_data/Wholesale_customers_data.csv"
    df = pd.read_csv(csv_path)
    X, y = pre_step(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_dim = X_train.shape[1]
    model = model_setting(X_train, y)
    weight_path = os.path.join(os.path.dirname(__file__), 'model_weight', 'ngp_mlp_weights.pth')
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.to(device)

    dataset = TabularDataset(X, y)
    batch_size = 128
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    methods = [
        'KernelShap',
        'Lime',
        'FeatureAblation',
        'Occlusion',
        'DeepLift',
        'Saliency',
        'GradientShap',
        'IntegratedGradients'
    ]
    print(f"device: {device}")
    print("Start explaining all methods...")
    for method in tqdm(methods, desc="Interpretation method"):
        print(f"\n{'='*50}")
        print(f"running {method}...")
        print(f"{'='*50}")
        attributions = explain_batch(
            data_loader, model, method=method, device=device, batch_size=batch_size
        )
        print(f"{method} Explanation completed: {attributions.shape}")

        save_root = '../../../data/generated_saliency/wcd/mlp'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        for i in range(attributions.shape[0]):
            sample_dir = os.path.join(save_root, f"sample_{i:05d}")
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
                df.iloc[[i]].to_csv(os.path.join(sample_dir, "data.csv"), index=False)
            attributions_path = os.path.join(sample_dir, "saliency")
            if not os.path.exists(attributions_path):
                os.makedirs(attributions_path)
            np.save(os.path.join(attributions_path, f"{method}.npy"), attributions[i])
            np.save(os.path.join(sample_dir, "input.npy"), X[i])


import os
import json
import shutil
import random

def build_saliency_dataset(data_dir, dataset, model, train_ratio=0.8):

    METHOD_NAME_MAP = {
        "integratedgradients": "IntegratedGradients",
        "gradientshap": "GradientShap",
        "saliency": "Saliency",
        "deeplift": "DeepLift",
        "occlusion": "Occlusion",
        "featureablation": "FeatureAblation",
        "lime": "Lime",
        "kernelshap": "KernelShap"
    }
    src_root = os.path.join(data_dir, dataset, model)
    
    sample_dirs = []
    for sample_name in os.listdir(src_root):
        sample_path = os.path.join(src_root, sample_name)
        if not os.path.isdir(sample_path):
            continue
        filter_json_path = os.path.join(sample_path, 'filter.json')
        original_img_path = os.path.join(sample_path, 'data.csv')
        numpy_dir = os.path.join(sample_path, 'saliency')
        if not (os.path.exists(filter_json_path) and os.path.exists(original_img_path) and os.path.isdir(numpy_dir)):
            continue
        sample_dirs.append(sample_name)
    
    # 随机划分训练集和测试集
    random.shuffle(sample_dirs)
    split_idx = int(len(sample_dirs) * train_ratio)
    train_samples = sample_dirs[:split_idx]
    test_samples = sample_dirs[split_idx:]

    # 处理训练集
    train_dst_root = f'../../data/explanation_dataset/{dataset}/{model}/train'
    train_sample_dir = os.path.join(train_dst_root, 'sample')
    train_saliency_dir = os.path.join(train_dst_root, 'saliency')
    os.makedirs(train_sample_dir, exist_ok=True)
    os.makedirs(train_saliency_dir, exist_ok=True)
    
    for sample_name in train_samples:
        sample_path = os.path.join(src_root, sample_name)
        filter_json_path = os.path.join(sample_path, 'filter.json')
        original_img_path = os.path.join(sample_path, 'data.csv')
        numpy_dir = os.path.join(sample_path, 'saliency')
        
        with open(filter_json_path, 'r') as f:
            methods = json.load(f)
        for method in methods:
            method_file = METHOD_NAME_MAP.get(method.lower(), method)
            # 1. 复制图片
            dst_img_name = f'{sample_name}_{method_file}.npy'
            dst_img_path = os.path.join(train_sample_dir, dst_img_name)
            shutil.copyfile(original_img_path, dst_img_path)
            # 2. 复制npy
            npy_src_path = os.path.join(numpy_dir, f'{method_file}.npy')
            dst_npy_name = f'{sample_name}_{method_file}.npy'
            dst_npy_path = os.path.join(train_saliency_dir, dst_npy_name)
            if os.path.exists(npy_src_path):
                shutil.copyfile(npy_src_path, dst_npy_path)

    test_dst_root = f'../../data/explanation_dataset/{dataset}/{model}/test'
    os.makedirs(test_dst_root, exist_ok=True)
    
    for sample_name in test_samples:
        src_sample_path = os.path.join(src_root, sample_name)
        dst_sample_path = os.path.join(test_dst_root, sample_name)

        if os.path.exists(dst_sample_path):
            shutil.rmtree(dst_sample_path)
        shutil.copytree(src_sample_path, dst_sample_path)

        

if __name__ == '__main__':
    data_dir = '/home/gyh/ryy/code/latec/data/generated_saliency'
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    dataset = 'wcd'
    model = 'mlp'

    build_saliency_dataset(data_dir, dataset, model) 
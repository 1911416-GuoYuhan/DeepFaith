import os
import json
import numpy as np

def filter(data_dir, dataset, model, p=0.2, ext='.npy'):


    saliency_dir = os.path.join(data_dir, dataset, model)
 
    dirs = os.listdir(saliency_dir)
    save_path = os.path.join(data_dir,'pvalue')

    for dir_name in dirs:
        dir_path = os.path.join(saliency_dir, dir_name)
        faith_json_path = os.path.join(dir_path, 'faithfulness.json')
        unique_json_path = os.path.join(dir_path, 'unique_images.json')
        filter_json_path = os.path.join(dir_path, 'filter.json')
        if not (os.path.exists(faith_json_path) and os.path.exists(unique_json_path)):
            continue
        with open(faith_json_path, 'r') as f:
            faith_data = json.load(f)
            # print(faith_data)
        metric_values = {}
        for method, metrics in faith_data.items():
            for metric, value in metrics.items():
                if metric not in metric_values:
                    metric_values[metric] = []
                metric_values[metric].append(value)

        p_quantiles = {}
        for metric, values in metric_values.items():
            if metric in ['deletion', 'positive']:
                p_quantiles[metric] = float(np.quantile(values, 1-p))
            else:
                p_quantiles[metric] = float(np.quantile(values, p))
     
        
   
        quantile_txt_path = os.path.join(save_path, model+'_'+dataset+'_'+'_quantiles.txt')
        with open(quantile_txt_path, 'a') as f:
            json.dump({dir_name: p_quantiles}, f, indent=2, ensure_ascii=False)
    
        with open(unique_json_path, 'r') as f:
            unique_images = json.load(f)
        method_names = [img.split(ext)[0] for img in unique_images if img.endswith(ext)]
 
        filtered_methods = []
        for method in method_names:
            method = method.lower()
            if method not in faith_data:
                continue
            metrics = faith_data[method]
            all_conditions_met = True
            for metric, value in metrics.items():
                if metric not in p_quantiles:
                    continue
                p_value = p_quantiles[metric]
                if metric in ['deletion', 'positive']:
                    if value > p_value:
                        all_conditions_met = False
                        break
                else:
                    if value < p_value:
                        all_conditions_met = False
                        break
            
            if all_conditions_met:
                filtered_methods.append(method)
            
        with open(filter_json_path, 'w') as f:
            json.dump(filtered_methods, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':

    data_dir = '../../data/generated_saliency'
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    dataset = 'oct'
    model = 'resnet'
    p = 0.13

    filter(data_dir, dataset, model, p, ext='.npy') 
import os
import json
import numpy as np

from image.generate_saliency import sub_dir_numpy, sub_dir_saliency

def cosine_similarity_np(vec1, vec2):

    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("dimension")
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(similarity)


def get_files_to_remove(duplicates):
    """
    Get a list of files to remove.

    Args:
        duplicates: A dictionary with file name as key and a list of duplicate file names as value.

    Returns:
        A list of files that should be removed.
    """
    # iterate over dict_ret keys, get value for the key and delete the dict keys that are in the value list
    files_to_remove = set()

    for k, v in duplicates.items():
        tmp = []
        for i in v:
            tmp.append(i)
        if k not in files_to_remove:
            files_to_remove.update(tmp)

    return list(files_to_remove)


def get_text_files_to_remove(text_saliencys,threshold=0.9):
    
    duplicates = {}
    for xai_method, saliency in text_saliencys.items():
        duplicates[xai_method] = []
        for xai_method_compare, saliency_compare in text_saliencys.items():
            if cosine_similarity_np(saliency,saliency_compare)>= threshold and xai_method != xai_method_compare:
                duplicates[xai_method].append(xai_method_compare)
    files_to_remove = get_files_to_remove(duplicates)
    return files_to_remove

def get_text_to_remove_by_dir(text_dir,threshold=0.9):
    text_saliencys = {}

    for saliency in os.listdir(text_dir):
        saliency_path = os.path.join(text_dir,saliency)
        text_saliencys[saliency] = np.load(saliency_path)
        # print(saliency,text_saliencys[saliency].shape)

    return get_text_files_to_remove(text_saliencys,threshold=threshold)


def saliency_dedup(data_dir,dataset,model):

    saliency_dir = os.path.join(data_dir,dataset,model)
    dirs = os.listdir(saliency_dir)

    for dir in dirs:
        text_dir = os.path.join(saliency_dir,dir,sub_dir_saliency)

        duplicates_list = get_text_to_remove_by_dir(text_dir,threshold=0.6)
        # print(dir,duplicates_list_cnn)
        texts = os.listdir(text_dir)
        unique_texts = []

        for text in texts:
            if (text not in duplicates_list) and ('origin' not in text):
                unique_texts.append(text)

        unique_json_path = os.path.join(saliency_dir, dir, 'unique_texts.json')
        with open(unique_json_path, 'w') as f:
            json.dump(unique_texts, f, indent=2)
        print('Sample',unique_json_path,'unique texts',len(unique_texts))
        # break


if __name__ == '__main__':
    data_dir = '../../data/generated_saliency'
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dataset = 'imdb'
    
    model = 'lstm'

    #saliency_dedup(data_dir,dataset,model,dedup_dir)
    saliency_dedup(data_dir,dataset,model)

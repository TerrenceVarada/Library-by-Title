import json
import torch
import logging
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import *
from utils.tools import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def embedding_mean_pooling(text):
    inputs = tokenizer(text, return_tensors="pt").to('mps')
    outputs = model(**inputs)
    embeddings = outputs.logits.mean(dim=1)

    return embeddings.cpu().detach().numpy()


def get_title_path(doc_folder, write_path):
    path_list = get_file_paths(doc_folder)
    info_lst = []
    for i in tqdm(range(len(path_list))):
        info = {}
        info['path'] = path_list[i]
        info['name'] = get_file_name(path_list[i])
        info_lst.append(info)

    with open(write_path, "w", encoding='utf-8') as f:
        f.write(json.dumps(info_lst, ensure_ascii=False))
    return info_lst


def combine_lists_to_dict(keys, values):
    dict_result = {}
    for i in range(len(keys)):
        dict_result[keys[i]] = values[i]
    return dict_result


def get_embedding(data, write_path):
    res_lst = []
    for i in tqdm(range(len(data))):
        info = data[i]
        title = info['name']
        question = f"用不多于200个字概括《{title}》这本书说了什么。"
        try:
            response, history = model.chat(tokenizer, question, history=[])
            info['response'] = response
            embedding = embedding_mean_pooling(response)
            info['embedding'] = json.dumps(embedding, cls=NumpyEncoder)
            res_lst.append(info)
        except Exception as e:
            logging.info(f"{info['path']} -- {e}")

        with open(write_path, "w", encoding='utf-8') as f:
            f.write(json.dumps(res_lst, ensure_ascii=False))


def parse_info(data):
    res = {'path': [], 'name': [], 'response': [], 'embedding': []}
    for i in tqdm(range(len(data))):
        _info = data[i]
        for k in _info.keys():
            if k == 'embedding':
                res['embedding'].append(json.loads(_info['embedding'])[0])
            else:
                res[k].append(_info[k])
    return res


def cluster_vectors(vectors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(vectors)
    labels = kmeans.fit_predict(vectors)
    return labels, kmeans


def get_best_k(vectors, max_cluster, min_cluster):
    scores = []
    labels_info = {}
    for k in tqdm(range(min_cluster, max_cluster)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=700)
        labels = kmeans.fit_predict(vectors)
        labels_info[k] = labels
        score = silhouette_score(vectors, labels)
        scores.append(score)
    best_k = np.argmax(scores) + 2
    logging.info(f'Best k: {best_k}')
    return labels_info[best_k], cluster_vectors(vectors, best_k)


def get_closest_indices(num_clusters, vectors, kmeans):
    closest_indices = []

    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)

    logging.info(len(selected_indices))
    return sorted(closest_indices)


def move_files(res, selected_indices, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    res = res[['path', 'labels']]
    selected_df = res.loc[selected_indices].sort_values(['labels'])
    selected_df['path'] = [get_file_name(x) for x in selected_df['path'].tolist()]
    selected_info = combine_lists_to_dict(selected_df['labels'].tolist(), selected_df['path'].tolist())
    for l in res['labels'].unique():
        _df = res[res['labels'] == l]
        try:
            sub_filder = f'{out_folder}/{l}-{get_file_name(selected_info[l])}'
        except:
            sub_filder = f'{out_folder}/{l}'
        if not os.path.exists(sub_filder):
            os.makedirs(sub_filder)
        for p in _df['path'].tolist():
            destination_file_path = f"{sub_filder}/{p.split('/')[-1]}"
            try:
                shutil.copy(p, destination_file_path)
            except:
                print(p, destination_file_path)


def get_cluster(data, max_clusters, cluster_search, write_path):
    res = parse_info(data)
    if cluster_search:
        labels, kmeans = get_best_k(res['embedding'], max_cluster=max_clusters, min_cluster=10)
    else:
        labels, kmeans = cluster_vectors(res['embedding'], num_clusters=max_clusters)
    res = pd.DataFrame(
        {'path': res['path'], 'name': res['name'], 'response': res['response'], 'labels': labels.tolist()})

    res.to_excel(write_path)
    print(res['labels'].value_counts())
    selected_indices = get_closest_indices(max_clusters, res['embedding'], kmeans)
    return selected_indices


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to('mps')

project_folder = 'Data'
cluster_search = True
max_clusters = 50

if __name__ == "__main__":
    data = get_title_path(doc_folder, write_path='doc_info.json')
    with open('doc_info.json', "r", encoding='utf-8') as f:
        data = json.load(f)
    get_embedding(data, write_path='res_info.json')
    with open('res_info.json', "r", encoding='utf-8') as f:
        data = json.load(f)
    res, selected_indices = get_cluster(data, max_clusters, cluster_search, write_path='res_info.xlsx')
    move_files(res, selected_indices, out_folder=new_library_path)

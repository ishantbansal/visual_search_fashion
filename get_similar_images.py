import torch
from net import f_model, c_model, p_model
from utils import FeatureExtractor
from torchvision import transforms
from PIL import Image
from scipy.spatial.distance import cdist
import numpy as np
import os , sys
import matplotlib.pyplot as plt
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_image', help='path to input image', required=True)
parser.add_argument('--model', help='path to model', required=True)
args = parser.parse_args()


GPU_ID = 0
CROP_SIZE = 480
num_similar_images = 10

DISTANCE_METRIC = ('cosine', 'cosine')
COLOR_WEIGHT = 0.1

model_path = args.model

input_image = args.input_image


if not os.path.isfile(input_image):
    print("check input image path")
    sys.exit()

main_model = f_model(model_path=model_path).cuda(GPU_ID)
color_model = c_model().cuda(GPU_ID)
pooling_model = p_model().cuda(GPU_ID)
extractor = FeatureExtractor(main_model, color_model, pooling_model)

data_transform_test = transforms.Compose([
    transforms.Resize(CROP_SIZE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_features(img_path):

    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')

    img = data_transform_test(img)
    img = img.unsqueeze(0)
    data = img.to('cuda')

    deep_feat, color_feat = extractor(data)

    deep_feat = deep_feat[0].squeeze()
    color_feat = color_feat[0]
    return deep_feat, color_feat

def get_top_n(dist, labels, retrieval_top_n):
    ind = np.argpartition(dist, -retrieval_top_n)[-retrieval_top_n:][::-1]
    ret = list(zip([labels[i] for i in ind], dist[ind]))
    ret = sorted(ret, key=lambda x: x[1], reverse=True)
    return ret


def get_similarity(feature, feats, metric='cosine'):
    dist = -cdist(np.expand_dims(feature, axis=0), feats, metric)[0]
    return dist


def get_deep_color_top_n(features, deep_feats, color_feats, labels, retrieval_top_n=5):
    deep_scores = get_similarity(features[0], deep_feats, DISTANCE_METRIC[0])
    color_scores = get_similarity(features[1], color_feats, DISTANCE_METRIC[1])
    results = get_top_n(deep_scores + color_scores * COLOR_WEIGHT, labels, retrieval_top_n)
    return results


def naive_query(features, deep_feats, color_feats, labels, retrieval_top_n=5):
    results = get_deep_color_top_n(features, deep_feats, color_feats, labels, retrieval_top_n)
    return results


DATASET_BASE = "./saved/"
def load_feat_db():
    feat_all = os.path.join(DATASET_BASE, 'all_feat.npy')
    feat_list = os.path.join(DATASET_BASE, 'all_feat.list')
    color_feat = os.path.join(DATASET_BASE, 'all_color_feat.npy')
    if not os.path.isfile(feat_list) or not os.path.isfile(feat_all) or not os.path.isfile(color_feat):
        print("No feature db file! Please run feature_extractor.py first.")
        return
    deep_feats = np.load(feat_all)
    color_feats = np.load(color_feat)
    with open(feat_list) as f:
        labels = list(map(lambda x: x.strip(), f.readlines()))
    return deep_feats, color_feats , labels


deep_feats, color_feats , labels = load_feat_db()
f = get_features(input_image)
result = naive_query(f, deep_feats, color_feats, labels, num_similar_images+1)

def visualize(original, result, cols=1):
    
    result = result[1:]
    n_images = len(result) + 1
    titles = ["Original"] + ["Score: {:.4f}".format(v) for k, v in result]
    images = [original] + [k for k, v in result]

    DATASET_BASE = './'

    mod_full_path = lambda x: os.path.join(DATASET_BASE, x) \
        if os.path.isfile(os.path.join(DATASET_BASE, x)) \
        else os.path.join(DATASET_BASE, x,)


    images = list(map(mod_full_path, images))
    images = list(map(lambda x: cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB), images))
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images * 0.25)
    plt.show()


visualize(input_image, result)

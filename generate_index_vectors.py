import torch
from net import f_model, c_model, p_model
from utils import FeatureExtractor
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
import os,sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='path to input dataset', required=True)
parser.add_argument('--model', help='path to model', required=True)
args = parser.parse_args()

GPU_ID = 0
CROP_SIZE = 480
model_path = args.model


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


def dump():

    deep_feats = []
    color_feats = []
    labels = []

    count = 0
    for img in glob.glob(args.path + "/*"):
            d1, c1 = get_features(img)
            deep_feats.append(d1)
            color_feats.append(c1)
            labels.append(img)
            count += 1
            print(count)


    DATASET_BASE = "./saved/"
    os.makedirs(DATASET_BASE, exist_ok=True)

    feat_all = os.path.join(DATASET_BASE, 'all_feat.npy')
    color_feat_all = os.path.join(DATASET_BASE, 'all_color_feat.npy')
    feat_list = os.path.join(DATASET_BASE, 'all_feat.list')

    with open(feat_list, "w") as fw:
        fw.write("\n".join(labels))
    np.save(feat_all, np.vstack(deep_feats))
    np.save(color_feat_all, np.vstack(color_feats))
    print("Dumped to all_feat.npy, all_color_feat.npy and all_feat.list.")


if __name__ == "__main__":
    dump()
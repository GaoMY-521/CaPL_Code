import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from datasets.imagenet import ImageNet

def pre_load_features(clip_model, loader, device, norm=True):

    features, labels = [], []

    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
                
            images, target = images.to(device), target.to(device)
            image_features = clip_model.encode_image(images)
            if norm:
                image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)

    features, labels = torch.cat(features), torch.cat(labels)
    
    return features, labels

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def main():
    torch.cuda.empty_cache()

    cfg = yaml.load(open('./configs/caltech101.yaml', 'r'), Loader=yaml.Loader)

    print('Loading CLIP model.')
    device = 'cuda:0'
    clip_model, preprocess = clip.load(cfg['backbone'], device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    
    cfg['subsample_classes'] = 'base'
    print("Preparing dataset.")
    dataset = build_dataset(cfg, cfg['dataset'], cfg['root_path'], cfg['shots'])
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    
    print('\nLoading visual features and labels from test set')
    test_features, test_labels = pre_load_features(clip_model, test_loader, device)
    
    # imagenet = ImageNet(cfg, cfg['root_path'], cfg['shots'], preprocess)
    # print("\nLoading visual features and labels from test set.")
    # test_features, test_labels = pre_load_features(clip_model, test_loader, device)

    cfg['subsample_classes'] = 'new'
    dataset_new = build_dataset(cfg, cfg['dataset'], cfg['root_path'], cfg['shots'])
    test_loader_new = build_data_loader(data_source=dataset_new.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

    print('\nLoading visual features and labels from new test set')
    test_features_new, test_labels_new = pre_load_features(clip_model, test_loader_new, device)
    
    # imagenet = ImageNet(cfg, cfg['root_path'], cfg['shots'], preprocess)
    # test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)
    # test_features_new, test_labels_new = pre_load_features(clip_model, test_loader, device)
    # test_labels_new = test_labels_new - 500
    
    print("\nGetting learned textual features.")
    text_features = torch.load(os.path.join("weights", f"{cfg['dataset']}.pt"), map_location='cuda:0')

    base_classnames = dataset.classnames
    new_classnames = dataset_new.classnames

    logits_new = 100. * test_features_new.float() @ text_features.T.float()[:, len(base_classnames):]
    new_acc = cls_acc(logits_new, test_labels_new)

    logits_base = 100. * test_features.float() @ text_features.T.float()[:, :len(base_classnames)]
    base_acc = cls_acc(logits_base, test_labels)

    H = 2 * base_acc * new_acc / (base_acc + new_acc)

    message = "base acc:\t%.2f  new acc:\t%.2f  H:\t%.2f \n" % (base_acc, new_acc, H)
    print(message)


if __name__ == '__main__':
    main()
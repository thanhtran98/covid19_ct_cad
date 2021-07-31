import os
import numpy as np
import random
import cv2
import torch
from torch.utils.data import Dataset

class CTLung_data(Dataset):
  def __init__(self, folders, transform=None):
    #"operation" can be 'resize' or 'crop'
    self.transform = transform
    dp = []
    for folder in folders:
      img_names = next(os.walk(folder))[2]
      img_paths = [os.path.join(folder, aa) for aa in img_names]
      dp += img_paths
    np.random.seed(131)
    random.shuffle(dp)
    self.folders_mask = [aa+'_mask' for aa in folders]
    self.paths = dp
    self.n_data = len(dp)
    print('total length data: {}'.format(self.n_data))
  def __len__(self):
    return self.n_data
  def __getitem__(self, idx):
    img_path = self.paths[idx]
    img = cv2.imread(img_path)
    # img = img[...,0:1]
    if self.transform:
      aug = self.transform(image = img)
      img = aug['image']
    return img, img_path

class COVID19_data(Dataset):
  def __init__(self, texts, folders, transform=None):
    #"operation" can be 'resize' or 'crop'
    self.transform = transform
    dp = []
    for i, text in enumerate(texts):
      with open(text, 'r') as fr:
        img_names = fr.read().split('\n')
        img_names = [aa for aa in img_names if aa != '']
        dp += [{'path':os.path.join(folders[i], aa), 'id':i} for aa in img_names]
    print(len(dp))
    np.random.seed(131)
    random.shuffle(dp)
    self.paths = dp
    self.n_data = len(dp)
  def __len__(self):
    return self.n_data
  def __getitem__(self, idx):
    pdict = self.paths[idx]
    img_path = pdict['path']
    id_ = float(pdict['id'])
    img = cv2.imread(img_path)
    if self.transform:
      img = self.transform(img)
    return img, torch.from_numpy(np.expand_dims(id_, axis=0).astype(np.float32))
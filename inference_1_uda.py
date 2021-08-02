####################
# Import Libraries
####################
import os
import sys
from PIL import Image
import cv2
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
import albumentations as A
import timm
from omegaconf import OmegaConf

import glob
from tqdm import tqdm

from sklearn.metrics import roc_auc_score

import torch


####################
# Utils
####################
def get_score(y_true, y_pred):
    score = roc_auc_score(y_true, y_pred)
    return score
  
def load_pytorch_model(ckpt_name, model, ignore_suffix='model'):
    state_dict = torch.load(ckpt_name, map_location='cpu')["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith(str(ignore_suffix)+"."):
            name = name.replace(str(ignore_suffix)+".", "", 1)  # remove `model.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model

def rot180(input: torch.Tensor) -> torch.Tensor:
    r"""Rotate a tensor image or a batch of tensor images
    180 degrees. Input must be a tensor of shape (C, H, W)
    or a batch of tensors :math:`(*, C, H, W)`.
    Args:
        input (torch.Tensor): input tensor
    Returns:
        torch.Tensor: The rotated image tensor
    """

    return torch.flip(input, [-2, -1])


def hflip(input: torch.Tensor) -> torch.Tensor:
    r"""Horizontally flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.
    Args:
        input (torch.Tensor): input tensor
    Returns:
        torch.Tensor: The horizontally flipped image tensor
    """
    w = input.shape[-1]
    return input[..., torch.arange(w - 1, -1, -1, device=input.device)]


def vflip(input: torch.Tensor) -> torch.Tensor:
    r"""Vertically flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.
    Args:
        input (torch.Tensor): input tensor
    Returns:
        torch.Tensor: The vertically flipped image tensor
    """

    h = input.shape[-2]
    return input[..., torch.arange(h - 1, -1, -1, device=input.device), :]
  
  
####################
# Config
####################

conf_dict = {'batch_size': 32,#32, 
             'epoch': 5,
             'height': 512,#640,
             'width': 512,
             'model_name': 'tf_efficientnet_b5_ns',
             'lr': 0.75e-3,
             'fold': 0,
             'drop_rate': 0.0,
             'drop_path_rate': 0.0,
             'data_dir': '/kqi/parent/22021621',
             'model_path': None,
             'output_dir': '/kqi/output/',
             'model_dir': '/kqi/parent/22021767',
             'snap': 2}
conf_base = OmegaConf.create(conf_dict)


####################
# Dataset
####################

class SETIDataset(Dataset):
    def __init__(self, df, transform=None, conf=None):
        self.df = df.reset_index(drop=True)
        self.labels = df['target'].values
        self.dir_names = df['dir'].values
        self.transform = transform
        self.conf = conf
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'id']
        file_path = os.path.join(self.dir_names[idx],"{}/{}.npy".format(img_id[0], img_id))
        
        image = np.load(file_path)
        image = image.astype(np.float32)
        image = np.vstack(image).transpose((1, 0))
        
        img_pl = Image.fromarray(image).resize((self.conf.height, self.conf.width), resample=Image.BICUBIC)
        image = np.array(img_pl)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = torch.from_numpy(image).unsqueeze(dim=0)

        label = torch.tensor([self.labels[idx]]).float()
        return image, label
      
 ####################
# Data Module
####################

class SETIDataModule(pl.LightningDataModule):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf  

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage=None):
        if stage == 'fit':
            df = pd.read_csv(os.path.join(self.conf.data_dir, "train_labels.csv"))
            df['dir'] = os.path.join(self.conf.data_dir, "train")
            
            # cv split
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
            for n, (train_index, val_index) in enumerate(skf.split(df, df['target'])):
                df.loc[val_index, 'fold'] = int(n)
            df['fold'] = df['fold'].astype(int)
            
            train_df = df[df['fold'] != self.conf.fold]
            valid_df = df[df['fold'] == self.conf.fold]
            
            train_transform = A.Compose([
                        A.VerticalFlip(p=0.5),
                        A.Cutout(max_h_size=int(self.conf.height * 0.1), max_w_size=int(self.conf.width * 0.1), num_holes=5, p=0.5),
                        ])
            self.train_dataset = SETIDataset(train_df, transform=train_transform,conf=self.conf)
            self.valid_dataset = SETIDataset(valid_df, transform=None, conf=self.conf)
            
        elif stage == 'test':
            test_df = pd.read_csv(os.path.join(self.conf.data_dir, "sample_submission.csv"))
            test_df['dir'] = os.path.join(self.conf.data_dir, "test")
            self.test_dataset = SETIDataset(test_df, transform=None, conf=self.conf)
         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)

# ====================================================
# Inference function
# ====================================================
def inference(models, test_loader):
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    with torch.no_grad():
        for i, (images) in tk0:
            images = images[0].cuda()
            avg_preds = []
            for model in models:
                y_preds = model(images)/2.0
                y_preds += model(vflip(images))/2.0
            
                avg_preds.append(y_preds.sigmoid().to('cpu').numpy())
            avg_preds = np.mean(avg_preds, axis=0)
            probs.append(avg_preds)
        probs = np.concatenate(probs)
    return probs
  
  
def gce(logits, target, q = 0.8):
    """ Generalized cross entropy.
    
    Reference: https://arxiv.org/abs/1805.07836
    """
    #probs = torch.nn.functional.softmax(logits, dim=1)
    probs = torch.sigmoid(logits)
    #probs_with_correct_idx = probs.index_select(-1, target).diag()
    #loss = (1. - probs**q) / q
    loss = 0.25 - (0.5 - probs)^2
    return loss.mean()

def adapt_batchnorm(model):
    model.eval()
    parameters = []
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            parameters.extend(module.parameters())
            module.train()
    return parameters
  
  
####################
# Train
####################  
def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    seed_everything(2021)

    #checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(conf.output_dir, 'ckpt/'), monitor='val_score', 
    #                                      save_last=True, save_top_k=5, mode='max', 
    #                                      save_weights_only=True, filename=f'fold{conf.fold}-'+'{epoch}-{val_score:.5f}')
    # get model path
    model_path = []
    for i in range(5):
        for j in range(conf.snap):
            target_model = glob.glob(os.path.join(conf.model_dir, f'fold{i}_{j}/ckpt/*epoch*.ckpt'))
            scores = [float(os.path.splitext(os.path.basename(i))[0].split('=')[-1]) for i in target_model]
            model_path.append(target_model[scores.index(max(scores))])

    data_module = SETIDataModule(conf)
    data_module.setup(stage='test')
    test_dataset = data_module.test_dataset
    test_loader =  DataLoader(test_dataset, batch_size=conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)
    
    for ckpt in model_path[:]:
        model = timm.create_model(model_name=conf.model_name, num_classes=1, pretrained=False, in_chans=1)
        model = load_pytorch_model(ckpt, model, ignore_suffix='model')
        model.cuda()
        parameters = adapt_batchnorm(model)
        #print(parameters)
        optimizer = torch.optim.SGD(model.parameters(), lr=conf.lr)
        #'''
        for images, _ in tqdm(test_loader, total=len(test_loader)):
            logits = model(images.cuda())
            #predictions = logits.argmax(dim = 1)
            predictions = torch.sigmoid(logits)#>0.5
            loss = gce(logits, predictions, q=0.8)
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #'''
        torch.save(model.to('cpu').state_dict(), os.path.join(conf.output_dir, os.path.basename(ckpt)))
        
    models = []
    for ckpt in model_path[:]:
        m = timm.create_model(model_name=conf.model_name, num_classes=1, pretrained=False, in_chans=1)
        #m = load_pytorch_model(os.path.join(conf.output_dir, os.path.basename(m_path)), m, ignore_suffix='model')
        m.load_state_dict(torch.load(os.path.join(conf.output_dir, os.path.basename(ckpt))))
        m.cuda()
        m.eval()
        models.append(m)
    
    predictions = inference(models, test_loader)
    
    test = pd.read_csv(os.path.join(conf.data_dir, "sample_submission.csv"))
    test['target'] = predictions
    test[['id', 'target']].to_csv(os.path.join(conf.output_dir, "submission.csv"), index=False)
    
    print(test[['id', 'target']].head())
    print(model_path)
    
    

if __name__ == "__main__":
    main()

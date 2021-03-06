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
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
import albumentations as A
import timm
from omegaconf import OmegaConf

from sklearn.metrics import roc_auc_score
####################
# Utils
####################
class CustomNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        #self.base_model.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten())
        in_features = self.base_model.num_features
        num_classes = self.base_model.num_classes
        drop_rate = self.base_model.drop_rate
        self.base_model.classifier = nn.Sequential(nn.Linear(in_features*6,512),nn.ReLU(),
                                                   nn.BatchNorm1d(512), nn.Dropout(drop_rate),
                                                   nn.Linear(512,num_classes))
    def forward(self, x):
        shape = x.shape # B,6,1,H,W
        x = x.view(-1,shape[2],shape[3],shape[4]) # Bx6,1,H,W
        #print(x.shape)
        x = self.base_model.forward_features(x)
        x = self.base_model.global_pool(x) # Bx6, N
        #print(x.shape)
        
        x = x.view(-1,6*self.base_model.num_features)
        #print(x.shape)
        
        if self.base_model.drop_rate > 0.:
            x = F.dropout(x, p=self.base_model.drop_rate, training=self.base_model.training)
        return self.base_model.classifier(x)
      
def get_score(y_true, y_pred):
    try:
        score = roc_auc_score(y_true, y_pred)
    except:
        score = 0.0
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
  
def add_signal(image):
    signal = np.zeros((273, 256))
    start_x, start_y = torch.randint(low=20, high=236, size=(2,)).numpy()
    length = torch.randint(low=1, high=19, size=(1,)).numpy()
    rad = torch.rand(1).numpy() * 2
    end_x = int(start_x + np.cos(rad * np.pi) * length)
    end_y = int(start_y + np.sin(rad * np.pi) * length)
    color = float(np.max(image)) * (np.random.rand()/2 + 0.5)
    thick = torch.randint(low=1, high=30, size=(1,)).numpy()
    signal = cv2.line(signal, (start_x, start_y), (end_x, end_y), color, thickness=1, lineType=cv2.LINE_AA)
    
    target = np.random.randint(0,6)
    image[target] = image[target] + signal
    label = 0
    if target%2==0:
        label = 1
    return image, label
  
####################
# Config
####################

conf_dict = {'batch_size': 8,#32, 
             'epoch': 30,
             'height': 512,#640,
             'width': 512,
             'model_name': 'efficientnet_b0',
             'lr': 0.001,
             'fold': 0,
             'drop_rate': 0.2,
             'drop_path_rate': 0.2,
             'data_dir': '../input/seti-breakthrough-listen',
             'model_path': None,
             'output_dir': './',
             'pseudo': 'pseudo802.csv',
             'low_th': 0.05,
             'high_th': 0.95,
             'trainer': {}}
conf_base = OmegaConf.create(conf_dict)

####################
# Dataset
####################

class SETIDataset(Dataset):
    def __init__(self, df, transform=None, conf=None, train=False):
        self.df = df.reset_index(drop=True)
        self.labels = df['target'].values
        self.dir_names = df['dir'].values
        self.transform = transform
        self.conf = conf
        self.train = train
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'id']
        file_path = os.path.join(self.dir_names[idx],"{}/{}.npy".format(img_id[0], img_id))
        
        image = np.load(file_path)
        image = image.astype(np.float32)

        label = self.labels[idx]
        if self.train:
            if label == 0:
                if torch.rand(1) < 0.20:
                    image, label = add_signal(image)
        label = torch.tensor([label]).float()
        
        for i in range(image.shape[0]):
            image[i] -= image[i].mean()
            image[i] /= image[i].std()
            
        #image = np.vstack(image).transpose((1, 0))
        image = image.transpose(0,2,1)
        
        #img_pl = Image.fromarray(image).resize((self.conf.height, self.conf.width), resample=Image.BICUBIC)
        #image = np.array(img_pl)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = torch.from_numpy(image).view(6,1,256,273) #.unsqueeze(dim=0)

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
            
            # old data
            '''
            old_train_df = pd.read_csv(os.path.join(self.conf.data_dir, "old_leaky_data/train_labels_old.csv"))
            old_test_df = pd.read_csv(os.path.join(self.conf.data_dir, "old_leaky_data/test_labels_old.csv"))
            old_train_df['dir'] = os.path.join(self.conf.data_dir, "old_leaky_data/train_old")
            old_test_df['dir'] = os.path.join(self.conf.data_dir, "old_leaky_data/test_old")
            train_df = pd.concat([train_df, old_train_df, old_test_df])
            '''
            # pseudo data
            '''
            pseudo_df = pd.read_csv(self.conf.pseudo)
            pseudo_df = pseudo_df[(pseudo_df['target']<self.conf.low_th)|(pseudo_df['target']>self.conf.high_th)]
            pseudo_df['dir'] = os.path.join(self.conf.data_dir, "test")
            train_df = pd.concat([train_df, pseudo_df])
            '''
            
            train_transform = A.Compose([
                        #A.Resize(height=self.conf.high, width=self.conf.width, interpolation=1), 
                        #A.Flip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        #A.HorizontalFlip(p=0.5),
                        #A.ShiftScaleRotate(p=0.5),
                        #A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
                        #A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
                        #A.CLAHE(clip_limit=(1,4), p=0.5),
                        #A.OneOf([
                        #    A.OpticalDistortion(distort_limit=1.0),
                        #    A.GridDistortion(num_steps=5, distort_limit=1.),
                        #    A.ElasticTransform(alpha=3),
                        #], p=0.20),
                        #A.OneOf([
                        #    A.GaussNoise(var_limit=[10, 50]),
                        #    A.GaussianBlur(),
                        #    A.MotionBlur(),
                        #    A.MedianBlur(),
                        #], p=0.20),
                        #A.Resize(size, size),
                        #A.OneOf([
                        #    A.JpegCompression(quality_lower=95, quality_upper=100, p=0.50),
                        #    A.Downscale(scale_min=0.75, scale_max=0.95),
                        #], p=0.2),
                        #A.IAAPiecewiseAffine(p=0.2),
                        #A.IAASharpen(p=0.2),
                        A.Cutout(max_h_size=int(self.conf.height * 0.1), max_w_size=int(self.conf.width * 0.1), num_holes=5, p=0.5),
                        #A.Normalize()
                        ])

            #valid_transform = A.Compose([
            #            A.Resize(height=self.conf.high, width=self.conf.width, interpolation=1), 
            #            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
            #            ])

            self.train_dataset = SETIDataset(train_df, transform=train_transform,conf=self.conf, train=True)
            self.valid_dataset = SETIDataset(valid_df, transform=None, conf=self.conf, train=False)
            
        elif stage == 'test':
            test_df = pd.read_csv(os.path.join(self.conf.data_dir, "sample_submission.csv"))
            test_df['dir'] = os.path.join(self.conf.data_dir, "test")
            test_transform = A.Compose([
                        A.Resize(height=self.conf.height, width=self.conf.width, interpolation=1, always_apply=False, p=1.0),
                        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                        ])
            self.test_dataset = SETIDataset(test_df, transform=test_transform, conf=self.conf)
         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)
        
####################
# Lightning Module
####################

class LitSystem(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        #self.conf = conf
        self.save_hyperparameters(conf)
        
        self.model = CustomNet(base_model=timm.create_model(model_name=self.hparams.model_name, num_classes=1, pretrained=True, in_chans=1,
                                       drop_rate=self.hparams.drop_rate, drop_path_rate=self.hparams.drop_path_rate))
        if self.hparams.model_path is not None:
            print(f'load model path: {self.hparams.model_path}')
            self.model = load_pytorch_model(self.hparams.model_path, self.model, ignore_suffix='model')
        
        self.criteria = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epoch)
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if self.current_epoch < self.hparams.epoch*0.8:
            # mixup
            alpha = 1.0
            lam = np.random.beta(alpha, alpha)
            batch_size = x.size()[0]
            index = torch.randperm(batch_size)
            x = lam * x + (1 - lam) * x[index, :]
            #y = lam * y +  (1 - lam) * y[index]
            y = y + y[index] - (y * y[index])
        
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y)
        
        return {
            "val_loss": loss,
            "y": y,
            "y_hat": y_hat
            }
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs]).cpu().detach().numpy()
        y_hat = torch.cat([x["y_hat"] for x in outputs]).cpu().detach().numpy()

        #preds = np.argmax(y_hat, axis=1)

        val_score = get_score(y, y_hat)

        self.log('avg_val_loss', avg_val_loss)
        self.log('val_score', val_score)

        
####################
# Train
####################  
def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    seed_everything(2021)

    tb_logger = loggers.TensorBoardLogger(save_dir=os.path.join(conf.output_dir, 'tb_log/'))
    csv_logger = loggers.CSVLogger(save_dir=os.path.join(conf.output_dir, 'csv_log/'))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(conf.output_dir, 'ckpt/'), monitor='val_score', 
                                          save_last=True, save_top_k=5, mode='max', 
                                          save_weights_only=True, filename=f'fold{conf.fold}-'+'{epoch}-{val_score:.5f}')

    data_module = SETIDataModule(conf)

    lit_model = LitSystem(conf)

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=conf.epoch,
        gpus=-1,
        amp_backend='native',
        amp_level='O2',
        precision=16,
        num_sanity_val_steps=10,
        val_check_interval=1.0,
        **conf.trainer
            )

    trainer.fit(lit_model, data_module)

if __name__ == "__main__":
    main()

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

from sklearn.metrics import roc_auc_score
####################
# Utils
####################
def get_score(y_true, y_pred):
    score = roc_auc_score(y_true, y_pred)
    return score
####################
# Config
####################

conf_dict = {'batch_size': 8,#32, 
             'epoch': 30,
             'high': 512,#640,
             'width': 512,
             'model_name': 'seresnet18',
             'lr': 0.001,
             'fold': 0,
             'drop_rate': 0.0,
             'drop_path_rate': 0.0,
             'data_dir': '../input/seti-breakthrough-listen',
             'output_dir': './'}
conf_base = OmegaConf.create(conf_dict)

####################
# Dataset
####################

class SETIDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.labels = df['target'].values
        self.dir_names = df['dir'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'id']
        file_path = os.path.join(self.dir_names[idx],"{}/{}.npy".format(img_id[0], img_id))
        
        image = np.load(file_path)
        image = image.astype(np.float32)
        bcd = image[[1,3,5]]
        aaa = image[[0,2,4]]
        bcd = np.vstack(bcd).transpose((1, 0)) # (1638, 256) -> (256, 1638)
        aaa = np.vstack(aaa).transpose((1, 0))
        image = np.asarray([aaa, bcd]).transpose(1,2,0)
        #print(image.shape)
        
        image = cv2.resize(image, (image.shape[0]*2, image.shape[0]*2), interpolation=cv2.INTER_CUBIC)
        #image = image.transpose(2,0,1)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = torch.from_numpy(image.transpose(2,0,1))#.unsqueeze(dim=0)

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
                        #A.Resize(height=self.conf.high, width=self.conf.width, interpolation=1), 
                        A.Flip(p=0.5),
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
                        A.Cutout(max_h_size=int(self.conf.high * 0.1), max_w_size=int(self.conf.width * 0.1), num_holes=5, p=0.5),
                        #A.Normalize()
                        ])

            #valid_transform = A.Compose([
            #            A.Resize(height=self.conf.high, width=self.conf.width, interpolation=1), 
            #            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
            #            ])

            self.train_dataset = SETIDataset(train_df, transform=train_transform)
            self.valid_dataset = SETIDataset(valid_df, transform=None)
            
        elif stage == 'test':
            test_df = pd.read_csv(os.path.join(self.conf.data_dir, "sample_submission.csv"))
            test_df['dir'] = os.path.join(self.conf.data_dir, "test")
            test_transform = A.Compose([
                        A.Resize(height=self.conf.high, width=self.conf.width, interpolation=1, always_apply=False, p=1.0),
                        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                        ])
            self.test_dataset = SETIDataset(test_df, transform=test_transform)
         
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
        self.model = timm.create_model(model_name=self.hparams.model_name, num_classes=1, pretrained=True, in_chans=2,
                                       drop_rate=self.hparams.drop_rate, drop_path_rate=self.hparams.drop_path_rate)
        state_dict = self.model.state_dict()
        self.model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.model.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        self.model.load_state_dict(state_dict, strict=True)
        
        self.criteria = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epoch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        
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
        val_check_interval=1.0
            )

    trainer.fit(lit_model, data_module)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import os

train_df = pd.read_csv('/kqi/parent/22021621/train_labels.csv')
test_df = pd.read_csv('pseudo802.csv')

train_df['dir'] = '/kqi/parent/22021621/train'
test_df['dir'] = '/kqi/parent/22021621/test'


idx = 1
img_id = train_df.loc[idx, 'id']
img_path = os.path.join(train_df.loc[idx, 'dir'],"{}/{}.npy".format(img_id[0], img_id))
image = np.load(img_path).astype(np.float32)
image = np.vstack(image).transpose((1, 0))
img_pl = Image.fromarray(image)#.resize((self.conf.height, self.conf.width), resample=Image.BICUBIC)
st.image(img_pl)

img_id = test_df.loc[idx, 'id']
img_path = os.path.join(test_df.loc[idx, 'dir'],"{}/{}.npy".format(img_id[0], img_id))
image = np.load(img_path).astype(np.float32)
image = np.vstack(image).transpose((1, 0))
img_pl = Image.fromarray(image)#.resize((self.conf.height, self.conf.width), resample=Image.BICUBIC)
st.image(img_pl)


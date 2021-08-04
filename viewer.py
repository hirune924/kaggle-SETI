import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

train_df = pd.read_csv('/kqi/parent/22021621/train_labels.csv')
test_df = pd.read_csv('pseudo802.csv')

train_df['dir'] = '/kqi/parent/22021621/train'
test_df['dir'] = '/kqi/parent/22021621/test'

pos = st.checkbox('neg or pos')
if pos:
  train_df = train_df[train_df['target']==1]
  test_df = test_df[test_df['target']>0.5]
else:
  train_df = train_df[train_df['target']==0]
  test_df = test_df[test_df['target']<0.5]
  
train_idx = st.slider('Train Index', min_value=0, max_value=len(train_df))
img_id = train_df.loc[train_idx, 'id']
st.text(f'TrainImage, id: {img_id}, idx: {train_idx}, target: {train_df.loc[train_idx, "target"]}')
img_path = os.path.join(train_df.loc[train_idx, 'dir'],"{}/{}.npy".format(img_id[0], img_id))
image = np.load(img_path).astype(np.float32)
image = np.vstack(image).transpose((1, 0))
fig, ax = plt.subplots()
ax.imshow(image)
st.pyplot(fig)
#img_pl = Image.fromarray(image)#.resize((self.conf.height, self.conf.width), resample=Image.BICUBIC)
#st.image(img_pl)


test_idx = st.slider('Train Index', min_value=0, max_value=len(test_df))
img_id = test_df.loc[test_idx, 'id']
st.text(f'TestImage, id: {img_id}, idx: {test_idx}, target: {test_df.loc[test_idx, "target"]}')
img_path = os.path.join(test_df.loc[test_idx, 'dir'],"{}/{}.npy".format(img_id[0], img_id))
image = np.load(img_path).astype(np.float32)
image = np.vstack(image).transpose((1, 0))
fig, ax = plt.subplots()
ax.imshow(image)
st.pyplot(fig)
#img_pl = Image.fromarray(image)#.resize((self.conf.height, self.conf.width), resample=Image.BICUBIC)
#st.image(img_pl)


python exp0023.py fold=0 batch_size=48 epoch=20 height=512 width=512 model_name=efficientnet_b4 drop_rate=0.2 drop_path_rate=0.2 \
data_dir=/kqi/parent/22020972 output_dir=/kqi/output/fold0_0

python exp0023.py fold=1 batch_size=48 epoch=20 height=512 width=512 model_name=efficientnet_b4 drop_rate=0.2 drop_path_rate=0.2 \
data_dir=/kqi/parent/22020972 output_dir=/kqi/output/fold1_0

python exp0023.py fold=2 batch_size=48 epoch=20 height=512 width=512 model_name=efficientnet_b4 drop_rate=0.2 drop_path_rate=0.2 \
data_dir=/kqi/parent/22020972 output_dir=/kqi/output/fold2_0

python exp0023.py fold=3 batch_size=48 epoch=20 height=512 width=512 model_name=efficientnet_b4 drop_rate=0.2 drop_path_rate=0.2 \
data_dir=/kqi/parent/22020972 output_dir=/kqi/output/fold3_0

python exp0023.py fold=4 batch_size=48 epoch=20 height=512 width=512 model_name=efficientnet_b4 drop_rate=0.2 drop_path_rate=0.2 \
data_dir=/kqi/parent/22020972 output_dir=/kqi/output/fold4_0



python exp0023.py fold=0 batch_size=48 epoch=20 height=512 width=512 model_name=efficientnet_b4 drop_rate=0.2 drop_path_rate=0.2 \
model_path=/kqi/output/fold0_0/ckpt/last.ckpt data_dir=/kqi/parent/22020972 output_dir=/kqi/output/fold0_1

python exp0023.py fold=1 batch_size=48 epoch=20 height=512 width=512 model_name=efficientnet_b4 drop_rate=0.2 drop_path_rate=0.2 \
model_path=/kqi/output/fold1_0/ckpt/last.ckpt data_dir=/kqi/parent/22020972 output_dir=/kqi/output/fold1_1

python exp0023.py fold=2 batch_size=48 epoch=20 height=512 width=512 model_name=efficientnet_b4 drop_rate=0.2 drop_path_rate=0.2 \
model_path=/kqi/output/fold2_0/ckpt/last.ckpt data_dir=/kqi/parent/22020972 output_dir=/kqi/output/fold2_1

python exp0023.py fold=3 batch_size=48 epoch=20 height=512 width=512 model_name=efficientnet_b4 drop_rate=0.2 drop_path_rate=0.2 \
model_path=/kqi/output/fold3_0/ckpt/last.ckpt data_dir=/kqi/parent/22020972 output_dir=/kqi/output/fold3_1

python exp0023.py fold=4 batch_size=48 epoch=20 height=512 width=512 model_name=efficientnet_b4 drop_rate=0.2 drop_path_rate=0.2 \
model_path=/kqi/output/fold4_0/ckpt/last.ckpt data_dir=/kqi/parent/22020972 output_dir=/kqi/output/fold4_1

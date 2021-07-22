#[aaa,bcd]の2段階cosine annealing, tf-efficientnet-b4-ns
#pip install -U timm

python baseline_1.py fold=0 batch_size=48 epoch=20 height=512 width=512 model_name=tf_efficientnet_b4_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/parent/22021628/fold0_1/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold0_2

python baseline_1.py fold=1 batch_size=48 epoch=20 height=512 width=512 model_name=tf_efficientnet_b4_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/parent/22021628/fold1_1/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold1_2

python baseline_1.py fold=2 batch_size=48 epoch=20 height=512 width=512 model_name=tf_efficientnet_b4_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/parent/22021628/fold2_1/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold2_2

python baseline_1.py fold=3 batch_size=48 epoch=20 height=512 width=512 model_name=tf_efficientnet_b4_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/parent/22021628/fold3_1/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold3_2

python baseline_1.py fold=4 batch_size=48 epoch=20 height=512 width=512 model_name=tf_efficientnet_b4_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/parent/22021628/fold4_1/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold4_2




python baseline_1.py fold=0 batch_size=48 epoch=20 height=512 width=512 model_name=tf_efficientnet_b4_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/output/fold0_2/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold0_3

python baseline_1.py fold=1 batch_size=48 epoch=20 height=512 width=512 model_name=tf_efficientnet_b4_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/output/fold1_2/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold1_3

python baseline_1.py fold=2 batch_size=48 epoch=20 height=512 width=512 model_name=tf_efficientnet_b4_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/output/fold2_2/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold2_3

python baseline_1.py fold=3 batch_size=48 epoch=20 height=512 width=512 model_name=tf_efficientnet_b4_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/output/fold3_2/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold3_3

python baseline_1.py fold=4 batch_size=48 epoch=20 height=512 width=512 model_name=tf_efficientnet_b4_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/output/fold4_2/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold4_3


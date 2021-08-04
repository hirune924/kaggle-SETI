#[aaa,bcd]の2段階cosine annealing, tf-efficientnet-b5-ns
#pip install -U timm

python baseline_1_ns.py fold=0 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=/kqi/parent/22021767/fold0_1/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold0_0

python baseline_1_ns.py fold=1 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=/kqi/parent/22021767/fold1_1/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold1_0

python baseline_1_ns.py fold=2 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=/kqi/parent/22021767/fold2_1/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold2_0

python baseline_1_ns.py fold=3 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=/kqi/parent/22021767/fold3_1/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold3_0

python baseline_1_ns.py fold=4 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=/kqi/parent/22021767/fold4_1/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold4_0

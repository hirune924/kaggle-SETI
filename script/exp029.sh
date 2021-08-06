#[aaa,bcd]の2段階cosine annealing, tf-efficientnet-b5-ns
#pip install -U timm

python baseline_1_pseudo_olddata.py fold=0 batch_size=36 epoch=15 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/parent/22021849/fold0_0/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold0_0 \
pseudo=pseudo807.csv low_th=0.04 high_th=0.9

python baseline_1_pseudo_olddata.py fold=1 batch_size=36 epoch=15 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/parent/22021849/fold1_0/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold1_0 \
pseudo=pseudo807.csv low_th=0.04 high_th=0.9

python baseline_1_pseudo_olddata.py fold=2 batch_size=36 epoch=15 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/parent/22021849/fold2_0/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold2_0 \
pseudo=pseudo807.csv low_th=0.04 high_th=0.9

python baseline_1_pseudo_olddata.py fold=3 batch_size=36 epoch=15 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/parent/22021849/fold3_0/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold3_0 \
pseudo=pseudo807.csv low_th=0.04 high_th=0.9

python baseline_1_pseudo_olddata.py fold=4 batch_size=36 epoch=15 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/parent/22021849/fold4_0/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold4_0 \
pseudo=pseudo807.csv low_th=0.04 high_th=0.9

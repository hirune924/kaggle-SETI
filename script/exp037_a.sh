#[aaa,bcd]の2段階cosine annealing, tf-efficientnet-b5-ns
#pip install -U timm

python baseline_1_pseudo.py fold=0 batch_size=64 epoch=20 height=512 width=512 model_name=tf_efficientnet_b6_ns drop_rate=0.5 drop_path_rate=0.2 \
model_path=/kqi/parent/22021801/fold0_1/ckpt/fold0-epoch=17-val_score=0.90077.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold0_0 \
pseudo=pseudo811.csv low_th=0.5 high_th=0.9999

python baseline_1_pseudo.py fold=1 batch_size=64 epoch=20 height=512 width=512 model_name=tf_efficientnet_b6_ns drop_rate=0.5 drop_path_rate=0.2 \
model_path=/kqi/parent/22021801/fold1_1/ckpt/fold1-epoch=12-val_score=0.89179.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold1_0 \
pseudo=pseudo811.csv low_th=0.5 high_th=0.9999

python baseline_1_pseudo.py fold=2 batch_size=64 epoch=20 height=512 width=512 model_name=tf_efficientnet_b6_ns drop_rate=0.5 drop_path_rate=0.2 \
model_path=/kqi/parent/22021801/fold2_1/ckpt/fold2-epoch=16-val_score=0.89582.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold2_0 \
pseudo=pseudo811.csv low_th=0.5 high_th=0.9999

python baseline_1_pseudo.py fold=3 batch_size=64 epoch=20 height=512 width=512 model_name=tf_efficientnet_b6_ns drop_rate=0.5 drop_path_rate=0.2 \
model_path=/kqi/parent/22021801/fold3_1/ckpt/fold3-epoch=15-val_score=0.90801.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold3_0 \
pseudo=pseudo811.csv low_th=0.5 high_th=0.9999

python baseline_1_pseudo.py fold=4 batch_size=64 epoch=20 height=512 width=512 model_name=tf_efficientnet_b6_ns drop_rate=0.5 drop_path_rate=0.2 \
model_path=/kqi/parent/22021801/fold4_1/ckpt/fold4-epoch=16-val_score=0.89180.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold4_0 \
pseudo=pseudo811.csv low_th=0.5 high_th=0.9999

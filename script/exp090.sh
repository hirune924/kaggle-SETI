#[aaa,bcd]の2段階cosine annealing, tf-efficientnet-b5-ns
#pip install -U timm

python baseline_1_ns_frac.py fold=0 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=/kqi/parent/22021886/fold0_0/ckpt/fold0-epoch=19-val_score=0.90587.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold0_0 \
pseudo=pseudo811.csv seed=${1}

python baseline_1_ns_frac.py fold=1 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=/kqi/parent/22021886/fold1_0/ckpt/fold1-epoch=19-val_score=0.90199.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold1_0 \
pseudo=pseudo811.csv seed=${1}

python baseline_1_ns_frac.py fold=2 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=/kqi/parent/22021886/fold2_0/ckpt/fold2-epoch=19-val_score=0.90302.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold2_0 \
pseudo=pseudo811.csv seed=${1}

python baseline_1_ns_frac.py fold=3 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=/kqi/parent/22021886/fold3_0/ckpt/fold3-epoch=18-val_score=0.91562.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold3_0 \
pseudo=pseudo811.csv seed=${1}

python baseline_1_ns_frac.py fold=4 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=/kqi/parent/22021886/fold4_0/ckpt/fold4-epoch=15-val_score=0.90265.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold4_0 \
pseudo=pseudo811.csv seed=${1}

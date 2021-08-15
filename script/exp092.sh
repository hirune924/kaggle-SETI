#[aaa,bcd]の2段階cosine annealing, tf-efficientnet-b5-ns
pip install -U timm

python baseline_1_ns_frac.py fold=0 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.5 drop_path_rate=0.3 \
model_path=/kqi/parent/22021766/fold0_1/ckpt/fold0-epoch=16-val_score=0.89814.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold0_0 \
pseudo=pseudo813.csv

python baseline_1_ns_frac.py fold=1 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.5 drop_path_rate=0.3 \
model_path=/kqi/parent/22021766/fold1_1/ckpt/fold1-epoch=17-val_score=0.89188.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold1_0 \
pseudo=pseudo813.csv

python baseline_1_ns_frac.py fold=2 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.5 drop_path_rate=0.3 \
model_path=/kqi/parent/22021766/fold2_1/ckpt/fold2-epoch=15-val_score=0.88993.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold2_0 \
pseudo=pseudo813.csv

python baseline_1_ns_frac.py fold=3 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.5 drop_path_rate=0.3 \
model_path=/kqi/parent/22021766/fold3_1/ckpt/fold3-epoch=17-val_score=0.90515.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold3_0 \
pseudo=pseudo813.csv

python baseline_1_ns_frac.py fold=4 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.5 drop_path_rate=0.3 \
model_path=/kqi/parent/22021766/fold4_1/ckpt/fold4-epoch=19-val_score=0.89206.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold4_0 \
pseudo=pseudo813.csv

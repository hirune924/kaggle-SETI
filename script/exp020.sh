#[aaa,bcd]の2段階cosine annealing, tf_efficientnet_b5_ns
pip install -U timm

python baseline_1_warmup_sep.py fold=0 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold0_0

python baseline_1_warmup_sep.py fold=1 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold1_0

python baseline_1_warmup_sep.py fold=2 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold2_0

python baseline_1_warmup_sep.py fold=3 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold3_0

python baseline_1_warmup_sep.py fold=4 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold4_0


#[aaa,bcd]の2段階cosine annealing, tf-efficientnet-b5-ns
pip install -U timm

python baseline_1_ns_frac_wu.py fold=1 batch_size=20 epoch=20 height=512 width=512 model_name=tf_efficientnet_b7_ns drop_rate=0.5 drop_path_rate=0.3 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold1_0 \
pseudo=pseudo813.csv seed=5656

python baseline_1_ns_frac_wu.py fold=2 batch_size=20 epoch=20 height=512 width=512 model_name=tf_efficientnet_b7_ns drop_rate=0.5 drop_path_rate=0.3 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold2_0 \
pseudo=pseudo813.csv seed=5656

python baseline_1_ns_frac_wu.py fold=3 batch_size=20 epoch=20 height=512 width=512 model_name=tf_efficientnet_b7_ns drop_rate=0.5 drop_path_rate=0.3 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold3_0 \
pseudo=pseudo813.csv seed=5656

python baseline_1_ns_frac_wu.py fold=4 batch_size=20 epoch=20 height=512 width=512 model_name=tf_efficientnet_b7_ns drop_rate=0.5 drop_path_rate=0.3 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold4_0 \
pseudo=pseudo813.csv seed=5656

python baseline_1_ns_frac_wu.py fold=0 batch_size=20 epoch=20 height=512 width=512 model_name=tf_efficientnet_b7_ns drop_rate=0.5 drop_path_rate=0.3 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold0_0 \
pseudo=pseudo813.csv seed=5656s
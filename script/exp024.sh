#[aaa,bcd]の2段階cosine annealing, dm_nfnet_f0
pip install -U timm

python baseline_1_warmup.py fold=0 batch_size=36 epoch=10 height=512 width=512 model_name=dm_nfnet_f0 drop_rate=0.4 drop_path_rate=0.2 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold0_0 lr=1e-4

python baseline_1_warmup.py fold=1 batch_size=36 epoch=10 height=512 width=512 model_name=dm_nfnet_f0 drop_rate=0.4 drop_path_rate=0.2 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold1_0 lr=1e-4

python baseline_1_warmup.py fold=2 batch_size=36 epoch=10 height=512 width=512 model_name=dm_nfnet_f0 drop_rate=0.4 drop_path_rate=0.2 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold2_0 lr=1e-4

python baseline_1_warmup.py fold=3 batch_size=36 epoch=10 height=512 width=512 model_name=dm_nfnet_f0 drop_rate=0.4 drop_path_rate=0.2 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold3_0 lr=1e-4

python baseline_1_warmup.py fold=4 batch_size=36 epoch=10 height=512 width=512 model_name=dm_nfnet_f0 drop_rate=0.4 drop_path_rate=0.2 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold4_0 lr=1e-4



python baseline_1_warmup.py fold=0 batch_size=36 epoch=10 height=512 width=512 model_name=dm_nfnet_f0 drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/output/fold0_0/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold0_1 lr=1e-4

python baseline_1_warmup.py fold=1 batch_size=36 epoch=10 height=512 width=512 model_name=dm_nfnet_f0 drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/output/fold1_0/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold1_1 lr=1e-4

python baseline_1_warmup.py fold=2 batch_size=36 epoch=10 height=512 width=512 model_name=dm_nfnet_f0 drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/output/fold2_0/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold2_1 lr=1e-4

python baseline_1_warmup.py fold=3 batch_size=36 epoch=10 height=512 width=512 model_name=dm_nfnet_f0 drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/output/fold3_0/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold3_1 lr=1e-4

python baseline_1_warmup.py fold=4 batch_size=36 epoch=10 height=512 width=512 model_name=dm_nfnet_f0 drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/output/fold4_0/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold4_1 lr=1e-4

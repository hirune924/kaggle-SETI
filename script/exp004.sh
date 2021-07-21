#[aaa,bcd]の2段階cosine annealing, tf-efficientnet-b4-ns
#pip install -U timm
python baseline_1.py fold=0 batch_size=12 epoch=20 height=512 width=512 model_name=ig_resnext101_32x32d drop_rate=0.4 drop_path_rate=0.2 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold0_0 trainer.accumulate_grad_batches=4

python baseline_1.py fold=1 batch_size=12 epoch=20 height=512 width=512 model_name=ig_resnext101_32x32d drop_rate=0.4 drop_path_rate=0.2 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold1_0 trainer.accumulate_grad_batches=4

python baseline_1.py fold=2 batch_size=12 epoch=20 height=512 width=512 model_name=ig_resnext101_32x32d drop_rate=0.4 drop_path_rate=0.2 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold2_0 trainer.accumulate_grad_batches=4

python baseline_1.py fold=3 batch_size=12 epoch=20 height=512 width=512 model_name=ig_resnext101_32x32d drop_rate=0.4 drop_path_rate=0.2 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold3_0 trainer.accumulate_grad_batches=4

python baseline_1.py fold=4 batch_size=12 epoch=20 height=512 width=512 model_name=ig_resnext101_32x32d drop_rate=0.4 drop_path_rate=0.2 \
data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold4_0 trainer.accumulate_grad_batches=4



python baseline_1.py fold=0 batch_size=12 epoch=20 height=512 width=512 model_name=ig_resnext101_32x32d drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/output/fold0_0/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold0_1 trainer.accumulate_grad_batches=4

python baseline_1.py fold=1 batch_size=12 epoch=20 height=512 width=512 model_name=ig_resnext101_32x32d drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/output/fold1_0/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold1_1 trainer.accumulate_grad_batches=4

python baseline_1.py fold=2 batch_size=12 epoch=20 height=512 width=512 model_name=ig_resnext101_32x32d drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/output/fold2_0/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold2_1 trainer.accumulate_grad_batches=4

python baseline_1.py fold=3 batch_size=12 epoch=20 height=512 width=512 model_name=ig_resnext101_32x32d drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/output/fold3_0/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold3_1 trainer.accumulate_grad_batches=4

python baseline_1.py fold=4 batch_size=12 epoch=20 height=512 width=512 model_name=ig_resnext101_32x32d drop_rate=0.4 drop_path_rate=0.2 \
model_path=/kqi/output/fold4_0/ckpt/last.ckpt data_dir=/kqi/parent/22021621 output_dir=/kqi/output/fold4_1 trainer.accumulate_grad_batches=4

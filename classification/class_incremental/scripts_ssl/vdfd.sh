# CIFAR-100
GPUID=0
OUTDIR=outputs/CIFAR100_10_10
REPEAT=1
mkdir -p $OUTDIR

python -u main_classinc_ssl.py  --agent_type vdfd_ssl_la --agent_name VDFDLA --hypermodel_type gcn --hypermodel_name GCN --dataset CIFAR100 --dataroot ../data --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --model_weight_decay 5e-4  --hypermodel_weight_decay 5e-4 --force_out_dim 100  --first_split_size 10 --other_split_size 10 --schedule 50 100  --batch_size 16 --model_name resnet18 --model_type resnet --model_lr 1e-4  --head_lr 5e-3 --hypermodel_lr 1e-4 --reg_coef 50 --epsilons 1e-3 --temp 1  --singular 1 | tee ${OUTDIR}/ssl_reg50_lr1e-4head_lr5e-3hplr1e-4_wd5e-4hpwd5e-4_bs16_epoch50100_eps1e-3.log


# TinyImageNet
GPUID=0
OUTDIR=outputs/Tiny_20_20
REPEAT=1
mkdir -p $OUTDIR

python -u main_classinc_ssl.py  --agent_type vdfd_ssl_la  --agent_name VDFDLA --hypermodel_type gcn --hypermodel_name GCN --dataset TinyImageNet --dataroot ../data/tiny-imagenet-200 --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --model_weight_decay 5e-4  --hypermodel_weight_decay 5e-4 --force_out_dim 200  --first_split_size 20 --other_split_size 20 --schedule 50 100  --batch_size 16 --model_name resnet18 --model_type resnet --model_lr 1e-4  --head_lr 2e-3 --hypermodel_lr 1e-4 --reg_coef 90 --epsilons 1e-3 --temp 1   --singular 1 | tee ${OUTDIR}/ssl_reg90_lr1e-4head_lr2e-3hplr1e-4_wd5e-4hpwd5e-4_bs16_epoch50100_eps1e-3.log
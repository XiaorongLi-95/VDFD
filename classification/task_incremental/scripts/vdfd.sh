#!/usr/bin/env bash
# 10-split CIFAR-100 
GPUID=0
OUTDIR=outputs/CIFAR100_10tasks
REPEAT=1
mkdir -p $OUTDIR
python -u main.py  --agent_type vdfd --agent_name VDFD --hypermodel_type gcn --hypermodel_name GCN --dataset CIFAR100 --dataroot ../../../data --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --model_weight_decay 5e-4  --hypermodel_weight_decay 5e-4 --force_out_dim 0  --first_split_size 10 --other_split_size 10 --schedule 30 60 80 --batch_size 32 --model_name resnet18 --model_type resnet --model_lr 1e-4  --head_lr 5e-3 --hypermodel_lr 7e-4 --reg_coef 10 --epsilons 1e-3 | tee ${OUTDIR}/reg10_lr1e-4head_lr5e-3hplr7e-4_wd5e-4hpwd5e-4_bs32_epoch306080_eps1e-3.log

# 20-split CIFAR-100 
GPUID=0
OUTDIR=outputs/CIFAR100_20tasks
REPEAT=1
mkdir -p $OUTDIR
python -u main.py  --agent_type vdfd --agent_name VDFD --hypermodel_type gcn --hypermodel_name GCN --dataset CIFAR100 --dataroot ../../../data --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --model_weight_decay 5e-4  --hypermodel_weight_decay 5e-4 --force_out_dim 0  --first_split_size 5 --other_split_size 5 --schedule 30 60 80 --batch_size 32 --model_name resnet18 --model_type resnet --model_lr 1e-4  --head_lr 5e-3 --hypermodel_lr 7e-4 --reg_coef 5 --epsilons 1e-3 | tee ${OUTDIR}/reg5_lr1e-4head_lr5e-3hplr7e-4_wd5e-4hpwd5e-4_bs32_epoch306080_eps1e-3.log

# 25-split TinyImageNet 
GPUID=0
REPEAT=1
OUTDIR=outputs/TinyImageNet_25tasks 
mkdir -p $OUTDIR

python -u main.py  --agent_type vdfd --agent_name VDFD --hypermodel_type gcn --hypermodel_name GCN --dataset TinyImageNet --dataroot ../../../data/tiny-imagenet-200/ --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --model_weight_decay 1e-5  --hypermodel_weight_decay 5e-5 --force_out_dim 0  --first_split_size 8 --other_split_size 8 --schedule 30 60 80 --batch_size 16 --model_name resnet18 --model_type resnet --model_lr 5e-5  --head_lr 5e-3 --hypermodel_lr 5e-4 --reg_coef 100 --epsilons 1e-3 | tee ${OUTDIR}/reg100_lr5e-5head_lr5e-3hplr5e-4_wd1e-5hpwd5e-5_bs16_epoch306080_eps1e-3.log

# 5-Datasets
GPUID=0
OUTDIR=outputs/5datasets
REPEAT=1
mkdir -p $OUTDIR

python -u main.py  --agent_type vdfd --agent_name VDFD --hypermodel_type gcn --hypermodel_name GCN --dataset FiveDatasets --dataroot ../../../data/fivedatasets --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --model_weight_decay 1e-4  --hypermodel_weight_decay 5e-5 --force_out_dim 0  --first_split_size 10 --other_split_size 10 --schedule 15 30  --batch_size 16 --model_name resnet18 --model_type resnet --model_lr 1e-4  --head_lr 5e-3 --hypermodel_lr 5e-4 --reg_coef 80 --epsilons 1e-3 | tee ${OUTDIR}/reg80_lr1e-4head_lr5e-3hplr5e-4_wd1e-4hpwd5e-5_bs16_epoch1530_eps1e-3.log



# 10-split SubImageNet with ResNet50 
GPUID=0
OUTDIR=outputs/SubImageNet_10tasks/resnet50
REPEAT=1
mkdir -p $OUTDIR

python -u main.py  --agent_type vdfd --agent_name VDFD --hypermodel_type gcn --hypermodel_name GCN --dataset SubImageNet --dataroot ../../../data/SubImageNet/ --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --model_weight_decay 1e-5 --hypermodel_weight_decay 1e-5 --force_out_dim 0  --first_split_size 10 --other_split_size 10 --schedule 30 60 80 --batch_size 16 --model_name resnet50 --model_type resnet --model_lr 1e-4  --head_lr 5e-3 --hypermodel_lr 1e-4 --reg_coef 200 --epsilons 1e-3 | tee ${OUTDIR}/reg200_lr1e-4head_lr5e-3hplr1e-4_wd1e-5hpwd1e-5_bs16_epoch306080_eps1e-3.log


# 10-split SubImageNet with SwinTransformer 
GPUID=0
OUTDIR=outputs/SubImageNet_10tasks/swin-t
REPEAT=1
mkdir -p $OUTDIR

python -u main_swin.py --agent_type vdfd_swin --agent_name VDFDSwin --hypermodel_type gcn --hypermodel_name GCN --dataset SubImageNet --dataroot ../../../data/SubImageNet/ --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --model_weight_decay 5e-5 --hypermodel_weight_decay 1e-5  --force_out_dim 0  --first_split_size 10 --other_split_size 10 --schedule 90 --warmup_epochs 5 --batch_size 16 --model_name SwinTransformer --model_type swin --model_lr 1e-4  --min_lr 1e-6 --warmup_lr 1e-7 --head_lr 1e-4 --hypermodel_lr 1e-4  --reg_coef 2000 --epsilons 1e-3   | tee ${OUTDIR}/reg2000_lr1e-4min1e-6warm1e-7_head_lr1e-4_hlr1e-4_wd5e-5hwd1e-5_bs16_COSepoch90-warm5_eps1e-3.log



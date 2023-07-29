# VDFD

PyTorch implementation of our TPAMI paper "Variational Data-Free Knowledge Distillation for Continual Learning"

**Title**: [Variational Data-Free Knowledge Distillation for Continual Learning](https://ieeexplore.ieee.org/document/10113287)

**Authors**: Xiaorong Li, Shipeng Wang, Jian Sun, Zongben Xu

**Email:** lixiaorong@stu.xjtu.edu.cn, lixiaorongxjtu@gmail.com

## Usage

### Task-incremental setting

- VDFD (w/o SSL)

```
cd classification/task_incremental
sh scripts/vdfd.sh
```

- VDFD

```
cd classification/task_incremental
sh scripts_ssl/vdfd.sh
```

#### Prepare Dataset

For 5-Dataset, you can download this dataset at [here](https://drive.google.com/file/d/1PeXFhrp4wgxLjlODL1p1VNgf4YY9coLu/view?usp=sharing).

### Class-incremental setting

- VDFD (w/o SSL)

```
cd classification/class_incremental
sh scripts/vdfd.sh
```

- VDFD

```
cd classification/class_incremental
sh scripts_ssl/vdfd.sh
```

### Domain-incremental setting

```
cd segmentation
sh scripts/vdfd.sh
```



## Requirements

### Task-incremental setting & Class-incremental setting

Python (3.6) 

PyTorch (1.8.0) 

timm

tensorboardX

tqdm

### Domain-incremental setting

Python (3.6)

Pytorch (1.8.1+cu102)

torchvision (0.9.1+cu102)

tensorboardX (1.8)

apex (0.1)

[inplace-abn](https://github.com/mapillary/inplace_abn) (1.0.7)

## Citation

```
@ARTICLE{Li_2023_tpami,
  author={Li, Xiaorong and Wang, Shipeng and Sun, Jian and Xu, Zongben},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Variational Data-Free Knowledge Distillation for Continual Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-17},
  doi={10.1109/TPAMI.2023.3271626}}
```

## Acknowledgment

The code of classification is based on [Adam-NSCL](https://github.com/ShipengWang/Adam-NSCL) and [Continual-Learning-Benchmark](https://github.com/GT-RIPL/Continual-Learning-Benchmark).

The code of segmentation is based on [PLOP](https://github.com/arthurdouillard/CVPR2021_PLOP) and [RCIL](https://github.com/zhangchbin/RCIL).
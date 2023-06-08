# 500-epoch training recipe with cosine decay 

# Cifar-100 examples to reproduce baseline, RandomAugment, Soft Augmentation, and RandAugment + Soft Augmentation experiments

# baseline
python train.py --task cifar100 --net resnet18 --da 0 --loss ce --ra 0 --pcrop 2 --crop 0

# RA
python train.py --task cifar100 --net resnet18 --da 0 --loss ce --ra 1 --pcrop 2 --crop 0

# SA
python train.py --task cifar100 --net resnet18 --da 2 --loss st --ra 0 --pcrop 2 --crop 12

# RA + SA
python train.py --task cifar100 --net resnet18 --da 2 --loss st --ra 1 --pcrop 2 --crop 12




<h1 align="center">
Soft Augmentation for Image Classification
</h1>

- Note:
  Due to the sudden fall of Argo AI that funded the research, I had to salvage the code and I'm running sanity checks with a single gaming GPU to make sure what gets released here reproduces the numbers in the paper. I plan to test and release code for ImageNet and self-supervised training models once I get my hands on some more GPUs. If you would like to help/collaborate you can find my email at the bottom.

Update 06/08/2023: Added code for Cifar experiments.

## Soft Augmentation

<p align="center">
    <img src="/imgs/tang.jpg" width="250"/>
    <img src="/imgs/sa_curve.png" width="250"/>
</p>
<p align="center">
    <b>Left:</b> Human visual classification performance from Tang et al. 2018. <b>Right:</b> Softening curves for soft crop augmentation.
</p>
The research is inspired by observations from human vision studies <a href="https://www.pnas.org/doi/abs/10.1073/pnas.1719397115" target="_blank">[Tang et al. 2018]</a>. Human visual recognition accuracy is not invariant but rather nonlinear; negligible drop for small occlusions, but large drop for large occlusions. We propose soft augmentation (SA) where the learning target softens non-linearly as a function of the degree of the transform applied to the sample.

<p align="center">
  <img src="/imgs/sa_fig1.png" width="750"/>
</p>

Traditional augmentation encourages invariance by requiring augmented samples to produce the same target label; we visualize the translational offset range (tx, ty) of Standard Hard Crop augmentations for 32 × 32 images from Cifar-100 on the <b>left</b>, reporting the top-1 error of a baseline ResNet-18. Naively increasing the augmentation range without reducing target confidence increases error (<b>middle</b>), but softening the target label by reducing the target confidence for extreme augmentations reduces the error (<b>right</b>), allowing for training with even more aggressive augmentations that may even produce blank images. Our work also shows that soft augmentations produce models that are more robust to occlusions (since they encounter larger occlusions during training) and models that are better calibrated (since they are trained to be less-confident on such occluded examples).    
## Getting started

Install `Pytorch`, `Tensorboard`, and other necessary libraries to get the environment set up.

# Supervised Cifar training

Here are Cifar-100 examples with ResNet-18 to reproduce baseline, RandomAugment, Soft Augmentation, and RandAugment + Soft Augmentation experiments. You can replace `resnet18` with other models like `resnet50`, `resnet101`, `wideresnet28`, `pyramidnet272`.

Navigate to cifar folder and run the following lines:

- Baseline

```
python train.py --task cifar100 --net resnet18 --da 0 --loss ce --ra 0
```

- RandAugment

```
python train.py --task cifar100 --net resnet18 --da 0 --loss ce --ra 1
```

- Soft Augmentation

```
python train.py --task cifar100 --net resnet18 --da 2 --loss st --ra 0 --pcrop 2 --crop 12
```

- RandAugment + Soft Augmentation

```
python train.py --task cifar100 --net resnet18 --da 2 --loss st --ra 1 --pcrop 2 --crop 12
```

## About this repository

This repository was built by <a href="https://scholar.google.com/citations?user=nVWQwHkAAAAJ&hl" target="_blank">Yang&nbsp;Liu</a> to accompany the following paper:

> [Soft Augmentation for Image Classification](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Soft_Augmentation_for_Image_Classification_CVPR_2023_paper.html).

If you find it useful, feel free to cite the paper:

```bibtex
@inproceedings{liu2023soft,
  title={Soft Augmentation for Image Classification},
  author={Liu, Yang and Yan, Shen and Leal-Taix{\'e}, Laura and Hays, James and Ramanan, Deva},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16241--16250},
  year={2023}
}
```

If something isn't clear or isn't working, let me know in the *Issues section* or contact [youngleoel@gmail.com](mailto:youngleoel@gmail.com).

## License

We are making our algorithm available under a [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. The other code we have used obeys other license restrictions as indicated in the subfolders.

""" helper function

original author baiyu
"""

from code import interact
import sys
import numpy
from numpy.core.numeric import zeros_like
from pandas.core.computation.ops import Op

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import cifar
import torch.nn.functional as F
from matplotlib import pyplot

import sys


def get_network(args):
    """ return given network
    """
    if args.task == 'cifar10':
        nclass = 10
    elif args.task == 'cifar100':
        nclass = 100

    if args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes=nclass)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_classes=nclass)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_classes=nclass)

    # Wide Resnet
    elif args.net == 'wideresnet28':
        from models.wideresnet import wideresnet28d0
        net = wideresnet28d0(num_classes=nclass)

    # PyramidNet
    elif args.net == 'pyramid272':
        from models.pyramidnet import PyramidNet
        net = PyramidNet('cifar', depth=272, alpha=200, 
                            num_classes=nclass, bottleneck=True)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class CIFAR10S(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            custom_transform: Optional[Callable] = None,
            transform_post: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR10S, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.custom_transform = custom_transform
        self.transform_post = transform_post
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.custom_transform is not None:
            img, target = self.custom_transform(img,target)
        
        if self.transform_post is not None:
            img = self.transform_post(img)
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

class CIFAR100S(CIFAR10S):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True,
                            task='cifar10',da=0,
                            max_p=1.0,
                            sigma_noise=0.0, pow_noise=1.0, bg_noise=1.0,
                            sigma_crop=10.0, pow_crop=4.0, bg_crop=1.0,
                            sigma_cut=0.0, pow_cut=4.0,
                            length_cut=16, mask_cut=1,
                            iou=False,
                            aa=0):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if task == 'cifar10':
        nclass = 10
    elif task == 'cifar100':
        nclass = 100
    t2 = None
    custom_transform = None
    if da == -1:
        print("no data augmentation!")
        t = [
        ]

    elif da == 0: # original augmentation
        print("standard data augmentation!")
        t = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    elif da == 1: # cutout will be appended to the end
        print("standard data augmentation with cutout!")
        t = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            
        ]
    elif da == 2:
        print("Randomflip + SoftCrop")
        t = [
            transforms.RandomHorizontalFlip(),
        ]
        
        custom_transform = SoftCrop(
                            n_class=nclass,
                            sigma_crop=sigma_crop, t_crop=1.0, 
                            max_p_crop=max_p, pow_crop=pow_crop, 
                            bg_crop=bg_crop,
                            iou=iou)
        

    if aa == 1:
        print('using Random Augmentation')
        t.append(transforms.RandAugment())

    
    t.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if da == 1:
        t.append(Cutout(n_holes=1,length=length_cut,mask=mask_cut))
        
    transform_train = transforms.Compose(t)
    if task == 'cifar100':
        cifar100_training = CIFAR100S(root='./data', train=True, download=True, 
                                      transform=transform_train,
                                      custom_transform=custom_transform,
                                      transform_post=t2)#,alpha=alpha)
        print('CIFAR 100 training set')
    elif task == 'cifar10':
        cifar100_training = CIFAR10S(root='./data', train=True, download=True, 
                                      transform=transform_train,
                                      custom_transform=custom_transform,
                                      transform_post=t2)#,alpha=alpha)
    
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,drop_last=False)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True,task="cifar100",train=False):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    if task == "cifar100":
        cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform_test)
        print('CIFAR 100 test set')
    elif task == "cifar10":
        cifar100_test = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform_test)
    
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def get_test_dataloader_crop(mean, std, batch_size=16, num_workers=2, shuffle=True,task="cifar100",train=False, crop=0):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=crop),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    if task == "cifar100":
        cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform_test)
        print('CIFAR 100 test set')
    elif task == "cifar10":
        cifar100_test = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform_test)
    
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def get_test_corrupt_dataloader(mean, std, batch_size=16, num_workers=2, 
                                shuffle=True,task="cifar100",train=False,
                                ):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """
    test_data={}

    test_data.data = np.load(task + base_path + corruption + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    
    return test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def ComputeProb(x, T=0.25, n_classes=10, max_prob=1.0, pow=2.0):
    max_prob = torch.clamp_min(torch.tensor(max_prob),1/n_classes)
    if T <=0:
        T = 1e-10

    if x > T:
        return max_prob
    elif x > 0:
        a = (max_prob - 1/float(n_classes))/(T**pow)
        return max_prob - a * (T-x) ** pow
    else:
        return np.ones_like(x) * 1/n_classes

def DecodeTargetProb(targets):
    '''
    Helper function, takes targets as input, splits it into GT classes and probability
    if a target is 7.2, then the GT class is 7 with probability 1 - 0.2 = 0.8.
    '''
    classes = targets.long()
    probs = 1 - (targets - classes)
    return classes, probs

def EncodeTargetProb(classes,probs=None):
    '''
    Helper function, takes GT classes and probabilities as input, 
    outputs a combined encoding with integer part encoding GT class
    and decimal part encoding 1-probability
    if the GT class for a sample is 7 with probability 0.8
    then target is 7 + (1- 0.8).
    caveat: input probability should be greater than 0
    otherwise the output class will be wrong
    '''
    if probs is None:
        return classes.float()
    else:
        return classes.float() + 1 - probs

# official cutout implementation
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, mask=0):
        self.n_holes = n_holes
        self.length = length
        self.mask = mask
        print("noise mask: ", str(self.mask))

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)
        dim1 = img.size(1)
        dim2 = img.size(2)
        
        # noise mix
        bg_n = torch.rand((3,dim1,dim2))
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            if self.mask:
                img[:,int(y1): int(y2), int(x1): int(x2)] = bg_n[:,int(y1): int(y2), int(x1): int(x2)]
            else:
                mask[int(y1): int(y2), int(x1): int(x2)] = 0.

                mask = torch.from_numpy(mask)
                mask = mask.expand_as(img)
                img = img * mask

        return img

class SoftCrop: 
    '''
    crop image
    
    '''
    def __init__(self, n_class=10,
                 sigma_crop=10, t_crop=1.0, max_p_crop=1.0, pow_crop=4.0, bg_crop=1.0,
                 iou=False):
        
        self.n_class = n_class
        self.chance = 1/n_class

        # crop parameters
        self.sigma_crop = sigma_crop
        self.t_crop = t_crop
        self.max_p_crop = max_p_crop
        self.pow_crop = pow_crop
        self.bg_crop = bg_crop

        self.iou = iou # if true, use IoU to compute r, else use IoForeground
        #for debugging
        self.flag = True

        print("use soft crop")
        print("sigma: ", self.sigma_crop, " T: ", self.t_crop, " Max P: ", self.max_p_crop,
                "bg: ", self.bg_crop, "power: ", self.pow_crop, "IoU: ", self.iou)

    def draw_offset(self, sigma=1, limit=24, n=100):
        # draw an integer from gaussian within +/- limit
        for d in range(n):
            x = torch.randn((1))*sigma
            if abs(x) <= limit:
                return int(x)
        return int(0)

    def __call__(self, image, label):

        dim1 = image.size(1)
        dim2 = image.size(2)

        # Soft Crop
        #bg = torch.randn((3,dim1*3,dim2*3)) * self.bg_crop # create a 3x by 3x sized noise background
        bg = torch.ones((3,dim1*3,dim2*3)) * self.bg_crop *  torch.randn((3,1,1))# create a 3x by 3x sized noise background
        bg[:,dim1:2*dim1,dim2:2*dim2] = image # put image at the center patch
        offset1 = self.draw_offset(self.sigma_crop,dim1)
        offset2 = self.draw_offset(self.sigma_crop,dim2)
        
        left = offset1 + dim1
        top = offset2 + dim2
        right = offset1 + dim1 * 2
        bottom = offset2 + dim2 * 2

        # number of pixels in orignal image kept after cropping alone
        intersection = (dim1 - abs(offset1))*(dim2 - abs(offset2))
        # proportion of original pixels left after cutout and cropping
        if self.iou:
            overlap = intersection / (dim1 * dim2 * 2 - intersection)
        else:
            overlap = intersection / (dim1 * dim2)
        # now the max prob can not be larger than prob_mix
        prob_crop = ComputeProb(overlap,
                                T=self.t_crop,
                                max_prob=self.max_p_crop,
                                pow=self.pow_crop,
                                n_classes=self.n_class)

        new_image = bg[:, left: right, top: bottom] # crop image
        new_label = label + 1 - prob_crop #max(prob_crop*prob_mix,self.chance)
        #print(new_label)
        return torch.tensor(new_image), torch.tensor(new_label)


# mixup code
POW = 1
@torch.no_grad()
def mixup_data(x, y, alpha=1.0, use_cuda=True, pow=POW, t=1.0, n_classes=1e6):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1 - 1e-8

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_int = y.long()

    y_prob = (1 - y + y_int)
    prob_a = ComputeProb(lam, max_prob=y_prob, pow=pow,T=t,n_classes=n_classes)
    prob_b = ComputeProb((1 - lam), max_prob=y_prob[index], pow=pow,T=t,n_classes=n_classes)

    y_a = torch.tensor(y_int + 1 - prob_a)
    y_b = torch.tensor(y_int[index] + 1 - prob_b)

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def soft_target(pred, 
                gold, 
                other=False, 
                distribute=True, 
                reweight=False,
                soften_one_hot=True,
                lr_correction=False):
    #print(gold)
    gold = gold.unsqueeze(1)
    target = gold.long()
    prob =  (1 - (gold - target))
    # print()
    weight = torch.clone(prob).float() if reweight else torch.ones_like(prob).float()
    if lr_correction:
        weight = weight / weight.mean()
    n_class = pred.size(1)
    #if we distribute 1-prob to other classes
    scatter_mul = 1.0 if distribute else 0.0
    if soften_one_hot:
        if not other: # if there is an other class       
            one_hot = (torch.ones_like(pred) * (1 - prob) * scatter_mul/ (n_class - 1)).float()
            one_hot.scatter_(dim=1, index=target, src=prob.float())
        else:
            one_hot = torch.zeros_like(pred)
            one_hot.scatter_(dim=1, index=torch.ones_like(target)*(n_class-1), src=(1-prob.float())* scatter_mul)
            one_hot.scatter_(dim=1, index=target, src=prob.float())
    else:
        one_hot = torch.zeros_like(pred)
        one_hot.scatter_(dim=1, index=target, src=torch.ones_like(target).float())

    log_prob = F.log_softmax(pred, dim=1)

    kl = weight * F.kl_div(input=log_prob.float(), target=one_hot.float(), reduction='none').sum(-1)
    return kl.mean()

def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1).mean()


def ECE(y_true, y_pred, num_bins=11):
    y_pred = torch.nn.functional.softmax(y_pred)
    #print(y_pred.shape)
    y_p = np.squeeze(y_pred.cpu().numpy())
    y_t = np.squeeze(y_true.cpu().numpy())
    pred_y = np.argmax(y_p, axis=-1)
    #print(pred_y)
    correct = (pred_y == y_t).astype(np.float32)
    prob_y = np.max(y_p, axis=-1)
    #print(prob_y.shape)
    #print(y_t.shape)
    bins = np.linspace(start=0, stop=1.0, num=num_bins)
    binned = np.digitize(prob_y, bins=bins, right=True)

    errors = np.zeros(num_bins)
    confs = np.zeros(num_bins)
    counts = np.zeros(num_bins)
    corrects = np.zeros(num_bins)
    accs = np.zeros(num_bins)
    o = 0
    for b in range(num_bins):
        mask = binned == b
        #if np.any(mask):
        count = np.sum(mask)
        counts[b] = count
        corrects[b] = np.sum(correct[mask])
        if count > 0:
            accs[b] = corrects[b] / counts[b]
            confs[b] = np.mean(prob_y[mask])
            errors[b] = np.abs( accs[b] - np.mean(prob_y[mask]) ) * counts[b]

    return np.sum(errors)/y_pred.shape[0], confs, accs, np.array(counts)

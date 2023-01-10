#!coding:utf-8
from collections import Counter
import numpy as np
import torchvision as tv
import torchvision.transforms as transforms
from utils.nir_utils import *
from utils.randAug import RandAugmentMC
from utils.data_utils import NO_LABEL
from utils.data_utils import TransformWeakStrong as wstwice,TransformBaseWeakStrong as bwstwice

load = {}
def register_dataset(dataset):
    def warpper(f):
        load[dataset] = f
        return f
    return warpper

def encode_label(label):
    return NO_LABEL* (label +1)

def decode_label(label):
    return NO_LABEL * label -1

def split_relabel_data(np_labs, labels, label_per_class,
                        num_classes):
    """ Return the labeled indexes and unlabeled_indexes
    """
    labeled_idxs = []
    unlabed_idxs = []
    for id in range(num_classes):
        indexes = np.where(np_labs==id)[0]
        np.random.shuffle(indexes)
        labeled_idxs.extend(indexes[:label_per_class])
        unlabed_idxs.extend(indexes[label_per_class:])
    np.random.shuffle(labeled_idxs)
    np.random.shuffle(unlabed_idxs)
    ## relabel dataset
    for idx in unlabed_idxs:
        labels[idx] = encode_label(labels[idx])

    return labeled_idxs, unlabed_idxs
     
def split(target,ratio):
    c=Counter(target)
    q=[(int(i),int(j*ratio)) for (i,j) in  c.items()]
    label_idx,unlabel_idx=[],[]
    for cls_ind,label_num in q:
        idx=np.where(target==cls_ind)[0]
        np.random.shuffle(idx)
        label_idx.extend(idx[:label_num])
        unlabel_idx.extend(idx[label_num:])
    np.random.shuffle(label_idx)
    np.random.shuffle(unlabel_idx)
    for idx in unlabel_idx:
        target[idx] = encode_label(target[idx])
    return label_idx,unlabel_idx

@register_dataset('cifar10')
def cifar10(n_labels, data_root='./data-local/cifar10/'):
    channel_stats = dict(mean = [0.4914, 0.4822, 0.4465],
                         std = [0.2023, 0.1994, 0.2010])
    train_transform = transforms.Compose([
        transforms.Pad(2, padding_mode='reflect'),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = tv.datasets.CIFAR10(data_root, train=True, download=True,
                                   transform=train_transform)
    evalset = tv.datasets.CIFAR10(data_root, train=False, download=True,
                                   transform=eval_transform)
    num_classes = 10
    label_per_class = n_labels // num_classes
    labeled_idxs, unlabed_idxs = split_relabel_data(
                                    np.array(trainset.train_labels),
                                    trainset.train_labels,
                                    label_per_class,
                                    num_classes)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'label_idxs': labeled_idxs,
        'unlab_idxs': unlabed_idxs,
        'num_classes': num_classes
    }

@register_dataset('wscifar10')
def wscifar10(n_labels, data_root='./data-local/cifar10/'):
    channel_stats = dict(mean = [0.4914, 0.4822, 0.4465],
                         std = [0.2023, 0.1994, 0.2010])
    weak = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(2, padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    strong = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(2, padding_mode='reflect'),
        transforms.RandomCrop(32),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    train_transform = wstwice(weak, strong)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = tv.datasets.CIFAR10(data_root, train=True, download=True,
                                   transform=train_transform)
    evalset = tv.datasets.CIFAR10(data_root, train=False, download=True,
                                   transform=eval_transform)
    num_classes = 10
    label_per_class = n_labels // num_classes
    labeled_idxs, unlabed_idxs = split_relabel_data(
                                    np.array(trainset.targets),
                                    trainset.targets,
                                    label_per_class,
                                    num_classes)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'label_idxs': labeled_idxs,
        'unlab_idxs': unlabed_idxs,
        'num_classes': num_classes
    }

@register_dataset('bwscifar10')
def bwscifar10(n_labels, data_root='./data-local/cifar10/'):
    channel_stats = dict(mean = [0.4914, 0.4822, 0.4465],
                         std = [0.2023, 0.1994, 0.2010])
    base=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(2, padding_mode='reflect'),
    ])
    weak = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(2, padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    strong = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(2, padding_mode='reflect'),
        transforms.RandomCrop(32),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    train_transform = bwstwice(base,weak, strong)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = tv.datasets.CIFAR10(data_root, train=True, download=True,
                                   transform=train_transform)
    evalset = tv.datasets.CIFAR10(data_root, train=False, download=True,
                                   transform=eval_transform)
    num_classes = 10
    label_per_class = n_labels // num_classes
    labeled_idxs, unlabed_idxs = split_relabel_data(
                                    np.array(trainset.targets),
                                    trainset.targets,
                                    label_per_class,
                                    num_classes)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'label_idxs': labeled_idxs,
        'unlab_idxs': unlabed_idxs,
        'num_classes': num_classes
    }

@register_dataset('cifar100')
def cifar100(n_labels, data_root='./data-local/cifar100/'):
    channel_stats = dict(mean = [0.5071, 0.4867, 0.4408],
                         std = [0.2675, 0.2565, 0.2761])
    train_transform = transforms.Compose([
        transforms.Pad(2, padding_mode='reflect'),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = tv.datasets.CIFAR100(data_root, train=True, download=True,
                                   transform=train_transform)
    evalset = tv.datasets.CIFAR100(data_root, train=False, download=True,
                                   transform=eval_transform)
    num_classes = 100
    label_per_class = n_labels // num_classes
    labeled_idxs, unlabed_idxs = split_relabel_data(
                                    np.array(trainset.train_labels),
                                    trainset.train_labels,
                                    label_per_class,
                                    num_classes)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'labeled_idxs': labeled_idxs,
        'unlabeled_idxs': unlabed_idxs,
        'num_classes': num_classes
    }

@register_dataset('cls50')
def cls50(label_ratio):

    weak = transforms.Compose([
        ToTensor(),
        RandAugmentNir(choice_num=1, max_value_ratio=0.2)
    ])
    strong = transforms.Compose([
        ToTensor(),
        RandAugmentNir(choice_num=2, max_value_ratio=1)
    ])
    train_transform = wstwice(weak, strong)
    eval_transform = transforms.Compose([
        ToTensor(),
    ])

    train_data,train_target=read_npy('train')
    eval_data,eval_target=read_npy('test')
    
    trainset = NIRDataset(train_data,train_target,train_transform,50)
    evalset = NIRDataset(eval_data,eval_target,eval_transform,50)

    labeled_idxs, unlabed_idxs = split(
                                    trainset.targets,
                                    label_ratio)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'label_idxs': labeled_idxs,
        'unlab_idxs': unlabed_idxs,
        'num_classes': 50
    }
    
@register_dataset('cls50i')
def cls50i(label_ratio):

    weak = transforms.Compose([
        ToTensor(),
        RandAugmentNir(choice_num=1, max_value_ratio=0.2)
    ])
    strong = transforms.Compose([
        ToTensor(),
        RandAugmentNir(choice_num=2, max_value_ratio=1)
    ])
    train_transform = wstwice(weak, strong)
    eval_transform = transforms.Compose([
        ToTensor(),
    ])

    train_data,train_target=read_npy('train')
    eval_data,eval_target=read_npy('test')
    
    trainset = NIRDataset(train_data,train_target,train_transform,50,0)
    evalset = NIRDataset(eval_data,eval_target,eval_transform,50,0)

    labeled_idxs, unlabed_idxs = split(
                                    trainset.targets,
                                    label_ratio)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'label_idxs': labeled_idxs,
        'unlab_idxs': unlabed_idxs,
        'num_classes': 50
    }
    
@register_dataset('cls50a')
def cls50a(label_ratio):

    weak = transforms.Compose([
        ToTensor(),
        RandAugmentNir(choice_num=1, max_value_ratio=0.2)
    ])
    strong = transforms.Compose([
        ToTensor(),
        RandAugmentNir(choice_num=2, max_value_ratio=1)
    ])
    train_transform = wstwice(weak, strong)
    eval_transform = transforms.Compose([
        ToTensor(),
    ])

    train_data,train_target=read_npy('train')
    eval_data,eval_target=read_npy('test')
    
    trainset = NIRDataset(train_data,train_target,train_transform,50,1)
    evalset = NIRDataset(eval_data,eval_target,eval_transform,50,1)

    labeled_idxs, unlabed_idxs = split(
                                    trainset.targets,
                                    label_ratio)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'label_idxs': labeled_idxs,
        'unlab_idxs': unlabed_idxs,
        'num_classes': 50
    }
    
@register_dataset('cls50r')
def cls50r(label_ratio):

    weak = transforms.Compose([
        ToTensor(),
        RandAugmentNir(choice_num=1, max_value_ratio=0.2)
    ])
    strong = transforms.Compose([
        ToTensor(),
        RandAugmentNir(choice_num=2, max_value_ratio=1)
    ])
    train_transform = wstwice(weak, strong)
    eval_transform = transforms.Compose([
        ToTensor(),
    ])

    train_data,train_target=read_npy('train')
    eval_data,eval_target=read_npy('test')
    
    trainset = NIRDataset(train_data,train_target,train_transform,50,2)
    evalset = NIRDataset(eval_data,eval_target,eval_transform,50,2)

    labeled_idxs, unlabed_idxs = split(
                                    trainset.targets,
                                    label_ratio)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'label_idxs': labeled_idxs,
        'unlab_idxs': unlabed_idxs,
        'num_classes': 50
    }
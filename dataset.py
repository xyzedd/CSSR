import json
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import methods.util as util
import os

UNKNOWN_LABEL = -1

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

cifar_mean = (0.5, 0.5, 0.5)
cifar_std = (0.25, 0.25, 0.25)

tiny_mean = (0.5, 0.5, 0.5)
tiny_std = (0.25, 0.25, 0.25)

svhn_mean = (0.5, 0.5, 0.5)
svhn_std = (0.25, 0.25, 0.25)

workers = 6
test_workers = 6
use_droplast = True
require_org_image = True
no_test_transform = False

# DATA_PATH = '/HOME/scz1838/run/data'
# TINYIMAGENET_PATH = DATA_PATH + '/tiny-imagenet-200/'
# LARGE_OOD_PATH = '/HOME/scz1838/run/largeoodds'
# IMAGENET_PATH = '/data/public/imagenet2012'
DATA_PATH = './resources'
TINYIMAGENET_PATH = DATA_PATH + '/tiny-imagenet-200/'
LARGE_OOD_PATH = DATA_PATH + '/largeoodds'
IMAGENET_PATH = DATA_PATH + '/imagenet2012'

for mypath in [DATA_PATH, TINYIMAGENET_PATH, LARGE_OOD_PATH, IMAGENET_PATH]:
    if not os.path.exists(mypath):
        os.makedirs(mypath)


class tinyimagenet_data(Dataset):

    def __init__(self, _type, transform):
        if _type == 'train':
            self.ds = datasets.ImageFolder(
                f'{TINYIMAGENET_PATH}/train/', transform=transform)
            self.labels = [self.ds.samples[i][1] for i in range(len(self.ds))]
        elif _type == 'test':
            tmp_ds = datasets.ImageFolder(
                f'{TINYIMAGENET_PATH}/train/', transform=transform)
            cls2idx = tmp_ds.class_to_idx
            self.ds = datasets.ImageFolder(
                f'{TINYIMAGENET_PATH}/val/', transform=transform)
            with open(f'{TINYIMAGENET_PATH}/val/val_annotations.txt', 'r') as f:
                file2cls = {}
                for line in f.readlines():
                    line = line.strip().split('\t')
                    file2cls[line[0]] = line[1]
            self.labels = []
            for i in range(len(self.ds)):
                filename = self.ds.samples[i][0].split('/')[-1]
                self.labels.append(cls2idx[file2cls[filename]])
            # print("test labels",self.labels)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx][0], self.labels[idx]


class Imagenet1000(Dataset):

    lab_cvt = None

    def __init__(self, istrain, transform):

        set = "train" if istrain else "val"
        self.ds = datasets.ImageFolder(
            f'{IMAGENET_PATH}/{set}/', transform=transform)
        self.labels = [self.ds.samples[i][1] for i in range(len(self.ds))]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


class LargeOODDataset(Dataset):

    def __init__(self, ds_name, transform) -> None:
        super().__init__()
        data_path = f'{LARGE_OOD_PATH}/{ds_name}/'
        self.ds = datasets.ImageFolder(data_path, transform=transform)
        self.labels = [-1] * len(self.ds)

    def __len__(self,):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]


class PartialDataset(Dataset):

    def __init__(self, knwon_ds, lab_keep=None, lab_cvt=None) -> None:
        super().__init__()
        self.known_ds = knwon_ds
        labels = knwon_ds.labels
        if lab_cvt is None:  # by default, identity mapping
            lab_cvt = [i for i in range(1999)]
        if lab_keep is None:  # by default, keep positive labels
            lab_keep = [x for x in lab_cvt if x > -1]
        keep = {x for x in lab_keep}
        self.sample_indexes = [i for i in range(
            len(knwon_ds)) if lab_cvt[labels[i]] in keep]
        self.labels = [lab_cvt[labels[i]]
                       for i in range(len(knwon_ds)) if lab_cvt[labels[i]] in keep]
        self.labrefl = lab_cvt

    def __len__(self) -> int:
        return len(self.sample_indexes)

    def __getitem__(self, index: int):
        inp, lb = self.known_ds[self.sample_indexes[index]]
        return inp, self.labrefl[lb], index


class UnionDataset(Dataset):

    def __init__(self, ds_list) -> None:
        super().__init__()
        self.dslist = ds_list
        self.totallen = sum([len(ds) for ds in ds_list])
        self.labels = []
        for x in ds_list:
            self.labels += x.labels

    def __len__(self) -> int:
        return self.totallen

    def __getitem__(self, index: int):
        orgindex = index
        for ds in self.dslist:
            if index < len(ds):
                a, b, c = ds[index]
                return a, b, orgindex
            index -= len(ds)
        return None


def gen_transform(mean, std, crop=False, toPIL=False, imgsize=32, testmode=False):
    t = []
    if toPIL:
        t.append(transforms.ToPILImage())
    if not testmode:
        return transforms.Compose(t)
    if crop:
        if imgsize > 200:
            t += [transforms.Resize(256), transforms.CenterCrop(imgsize)]
        else:
            t.append(transforms.CenterCrop(imgsize))
    # print(t)
    # transforms.Compose : Composes several transforms together
    # Here ToTensor and Normalize is used
    return transforms.Compose(t + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def gen_cifar_transform(crop=False, toPIL=False, testmode=False):
    """A function/transform that takes in an PIL image and returns a transformed version. Eg. transforms.RandomCrop

    Args:
        crop (bool, optional): _description_. Defaults to False.
        toPIL (bool, optional): _description_. Defaults to False.
        testmode (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    return gen_transform(cifar_mean, cifar_std, crop, toPIL=toPIL, imgsize=32, testmode=testmode)


def gen_tinyimagenet_transform(crop=False, testmode=False):
    return gen_transform(tiny_mean, tiny_std, crop, False, imgsize=64, testmode=testmode)


def gen_imagenet_transform(crop=False, testmode=False):
    return gen_transform(imagenet_mean, imagenet_std, crop, False, imgsize=224, testmode=testmode)


def gen_svhn_transform(crop=False, toPIL=False, testmode=False):
    return gen_transform(svhn_mean, svhn_std, crop, toPIL=toPIL, imgsize=32, testmode=testmode)


def get_cifar10(settype):
    """See details at: https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html

    Args:
        settype (_type_): _description_

    Returns:
        _type_: _description_
    """
    if settype == 'train':
        trans = gen_cifar_transform()   # Calling the transform function
        ds = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=True, download=True, transform=trans)
    else:
        ds = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=False, download=True, transform=gen_cifar_transform(testmode=True))
    ds.labels = ds.targets
    return ds


def get_cifar100(settype):
    if settype == 'train':
        trans = gen_cifar_transform()
        ds = torchvision.datasets.CIFAR100(
            root=DATA_PATH, train=True, download=True, transform=trans)
    else:
        ds = torchvision.datasets.CIFAR100(
            root=DATA_PATH, train=False, download=True, transform=gen_cifar_transform(testmode=True))
    ds.labels = ds.targets
    return ds


def get_svhn(settype):
    if settype == 'train':
        trans = gen_svhn_transform()
        ds = torchvision.datasets.SVHN(
            root=DATA_PATH, split='train', download=True, transform=trans)
    else:
        ds = torchvision.datasets.SVHN(
            root=DATA_PATH, split='test', download=True, transform=gen_svhn_transform(testmode=True))
    return ds


def get_tinyimagenet(settype):
    if settype == 'train':
        trans = gen_tinyimagenet_transform()
        ds = tinyimagenet_data('train', trans)
    else:
        ds = tinyimagenet_data(
            'test', gen_tinyimagenet_transform(testmode=True))
    return ds


def get_imagenet1000(settype):
    if settype == 'train':
        trans = gen_imagenet_transform()
        ds = Imagenet1000(True, trans)
    else:
        ds = Imagenet1000(False, gen_imagenet_transform(
            crop=True, testmode=True))
    return ds


def get_ood_inaturalist(settype):
    if settype == 'train':
        raise Exception("OOD iNaturalist cannot be used as train set.")
    else:
        return LargeOODDataset('iNaturalist', gen_imagenet_transform(crop=True, testmode=True))


ds_dict = {
    "cifarova": get_cifar10,
    "cifar10": get_cifar10,
    "cifar100": get_cifar100,
    "svhn": get_svhn,
    "tinyimagenet": get_tinyimagenet,
    "imagenet": get_imagenet1000,
    'oodinaturalist': get_ood_inaturalist,
}

cache_base_ds = {

}


def get_ds_with_name(settype, ds_name):
    global cache_base_ds
    key = str(settype) + ds_name
    if key not in cache_base_ds.keys():
        cache_base_ds[key] = ds_dict[ds_name](settype)
    return cache_base_ds[key]


def get_partialds_with_name(settype, ds_name, label_cvt, label_keep):
    ds = get_ds_with_name(settype, ds_name)
    return PartialDataset(ds, label_keep, label_cvt)

# setting list [[ds_name, sample partition list, label convertion list],...]


def get_combined_dataset(settype, setting_list):
    ds_list = []
    for setting in setting_list:
        ds = get_partialds_with_name(
            settype, setting['dataset'], setting['convert_class'], setting['keep_class'])
        if ds.__len__() > 0:
            ds_list.append(ds)
    return UnionDataset(ds_list) if len(ds_list) > 0 else None


def get_combined_dataloaders(args, settings):
    istrain_mode = True
    print("Load with train mode :", istrain_mode)
    train_labeled = get_combined_dataset('train', settings['train'])
    test = get_combined_dataset('test', settings['test'])
    return torch.utils.data.DataLoader(train_labeled, batch_size=args.bs, shuffle=istrain_mode, num_workers=workers, pin_memory=True, drop_last=use_droplast) if train_labeled is not None else None,\
        torch.utils.data.DataLoader(test, batch_size=args.bs, shuffle=False, num_workers=test_workers,
                                    pin_memory=args.gpu != 'cpu') if test is not None else None


ds_classnum_dict = {
    'cifar10': 6,
    'svhn': 6,
    'tinyimagenet': 20,
    "imagenet": 1000,
}

imgsize_dict = {
    'cifar10': 32,
    'svhn': 32,
    'tinyimagenet': 64,
    "imagenet": 224,
}


def load_partitioned_dataset(args, ds):
    with open(ds, 'r') as f:
        settings = json.load(f)
    util.img_size = imgsize_dict[settings['name']]
    a, b = get_combined_dataloaders(args, settings)
    return a, b, ds_classnum_dict[settings['name']]

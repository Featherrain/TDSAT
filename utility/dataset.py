import os

import torch
import torch.utils.data as data
import numpy as np
from scipy.io import loadmat
from torchnet.dataset import TransformDataset, SplitDataset
from .util_dataset import worker_init_fn
import hdf5storage

class LoadMatKey(object):
    def __init__(self, key):
        self.key = key

    def __call__(self, mat):
        item = mat[self.key][:].transpose((2, 0, 1))
        return item.astype(np.float32)

class HSI2Tensor(object):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """
    def __init__(self, use_2dconv):
        self.use_2dconv = use_2dconv

    def __call__(self, hsi):
        if self.use_2dconv:
            img = torch.from_numpy(hsi)
        else:
            img = torch.from_numpy(hsi[None])
        return img.float()
    # 这是一个 Python 类 HSI2Tensor，实现了将高光谱图像(HSI)从 numpy 数组转换为 torch 张量(tensor)的功能。该类实现了 __init__ 和 __call__ 两个方法。
    #
    # 在 __init__ 方法中，接收一个布尔类型参数 use_2dconv，用于确定张量的维度。如果 use_2dconv 为 True，则生成的张量形状为 (C, H, W)，其中 C 是通道数，H 和 W 分别是高和宽；否则，生成的张量形状为 (1, C, H, W)。
    #
    # 在 __call__ 方法中，接收一个 numpy 数组 hsi，然后根据 use_2dconv 的值，生成对应形状的 torch 张量，并转换为 float 类型，最后返回该张量。如果 use_2dconv 为 True，则直接将 hsi 转换成形状为 (C, H, W) 的张量；否则，在通道维度上增加一个长度为 1 的维度，即使 hsi 形状为 (C, H, W) 的数组变成了形状为 (1, C, H, W) 的张量。
    #
    # 该类适用于图像分类任务或卷积神经网络中使用的数据读取器。根据输入张量的维度不同，数据读取器的代码可能会略有不同。

class LMDBDataset(data.Dataset):
    def __init__(self, db_path, repeat=1):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        self.repeat = repeat


    def __getitem__(self, index):
        index = index % self.length
        env = self.env
        with env.begin(write=False) as txn:
            raw_datum = txn.get('{:08}'.format(index).encode('ascii'))
        import caffe
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)

        flat_x = np.fromstring(datum.data, dtype=np.float32)
        x = flat_x.reshape(datum.channels, datum.height, datum.width)

        return x

    def __len__(self):
        return self.length * self.repeat

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

class NPYDataset(data.Dataset):
    def __init__(self, db_path, repeat=1):
        self.db_path = db_path
        self.datasets = np.load(db_path)
        self.length = len(self.datasets.files)
        self.repeat = repeat

    def __getitem__(self, index):
        index = index % self.length
        try:
            x = self.datasets['{:08}'.format(index)]
        except:
            x = self.datasets[bytes('{:08}'.format(index).encode())]
        return x

    def __len__(self):
        return self.length * self.repeat

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

class ImageTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, target_transform=None):
        super(ImageTransformDataset, self).__init__()

        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.dataset[idx]
        target = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def make_dataset(opt, train_transform, target_transform, common_transform, batch_size=None, repeat=1):
    # 这是一个用于构建数据集的函数，接受一些参数，返回一个数据加载器。
    #
    # 首先，通过opt.dataroot获取数据集文件的路径，并根据文件类型（后缀）使用不同的数据集类进行加载。如果文件类型为.db，则使用LMDBDataset加载，如果文件类型为.npz，则使用NPYDataset加载；否则抛出“file type not supported”的异常。
    #
    # 接着，将加载的数据集对象通过TransformDataset进行处理，即对数据集应用了一组共同的变换操作common_transform，这里的common_transform与前面提到的common_transform_1可能是相同的或者类似的。处理后得到的数据集对象赋值给dataset。
    #
    # 然后，调用ImageTransformDataset对原始数据集进行进一步处理，即对数据集的每个样本分别应用train_transform和target_transform。其中train_transform表示对输入图像应用的变换操作，target_transform表示对输出图像应用的变换操作。处理后得到的数据集对象赋值给train_dataset。
    #
    # 最后，利用torch.utils.data.DataLoader构建数据加载器train_loader，并返回该加载器。在配置DataLoader时，batch_size为可选参数，如果未指定则使用opt.batchSize；shuffle参数表示是否对数据集进行打乱；num_workers表示数据加载的并行线程数；pin_memory表示是否将GPU内存锁定，以加速数据传输；worker_init_fn表示每个worker线程初始化的回调函数，这里使用了worker_init_fn函数作为回调函数。
    suffix = opt.dataroot.split('.')[-1]
    if suffix == 'db':
        dataset = LMDBDataset(opt.dataroot, repeat=repeat)
    elif suffix == 'npz':
        dataset = NPYDataset(opt.dataroot, repeat=repeat)
    else:
        raise 'file type not supported'
    """Split patches dataset into training, validation parts"""
    dataset = TransformDataset(dataset, common_transform)# 对数据集应用了一组共同的变换操作common_transform

    train_dataset = ImageTransformDataset(dataset, train_transform, target_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                              batch_size=batch_size or opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)

    return train_loader


class MatDataFromFolder(torch.utils.data.Dataset):
    """Wrap mat data from folder"""
    def __init__(self, data_dir, load=loadmat, suffix='mat', fns=None, size=None):
        super(MatDataFromFolder, self).__init__()
        if fns is not None:
            self.filenames = [
                os.path.join(data_dir, fn) for fn in fns
            ]
        else:
            self.filenames = [
                os.path.join(data_dir, fn)
                for fn in os.listdir(data_dir)
                if fn.endswith(suffix)
            ]

        #self.load = load
        import hdf5storage#######################
        self.load = hdf5storage.loadmat
        if size and size <= len(self.filenames):
            self.filenames = self.filenames[:size]

    def __getitem__(self, index):


        mat = self.load(self.filenames[index])
        # mat = self.hdf5storage.loadmat(self.filenames[index])

        return mat

    def __len__(self):
        return len(self.filenames)


class LoadMatHSI(object):
    def __init__(self, input_key, gt_key, transform=None):
        self.gt_key = gt_key
        self.input_key = input_key
        self.transform = transform

    def __call__(self, mat):
        if self.transform:
            input = self.transform(mat[self.input_key][:].transpose((2, 0, 1)))
            gt = self.transform(mat[self.gt_key][:].transpose((2, 0, 1)))
        else:
            input = mat[self.input_key][:].transpose((2, 0, 1))
            gt = mat[self.gt_key][:].transpose((2, 0, 1))
        input = torch.from_numpy(input).float()
        gt = torch.from_numpy(gt).float()

        return input, gt
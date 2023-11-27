import shutil
import h5py
import os
from scipy.ndimage import zoom
from scipy.io import loadmat
import numpy as np
from utility import *
import math


def creat_train_val_test(path, train_size, val_size, test_size):
    fns = os.listdir(os.path.join(path, 'filefolder'))
    train_fns = fns[:math.floor(train_size*len(fns))]
    val_fns = fns[math.floor(train_size * len(fns)):math.floor((train_size+val_size) * len(fns))]
    test_fns = fns[math.floor((train_size+val_size) * len(fns)):math.floor((train_size+val_size+test_size) * len(fns))]

    basedir=path

    with open(os.path.join(basedir, 'train_fns.txt'), 'w') as f:
        for fn in train_fns:
            f.write(fn)
            f.write('\n')
    with open(os.path.join(basedir, 'validation_fns_gauss.txt'), 'w') as f:
        for fn in val_fns:
            f.write(fn)
            f.write('\n')
    with open(os.path.join(basedir, 'test_fns_gauss.txt'), 'w') as f:
        for fn in test_fns:
            f.write(fn)
            f.write('\n')
    with open(os.path.join(basedir, 'validation_fns_complex.txt'), 'w') as f:
        for fn in val_fns:
            f.write(fn)
            f.write('\n')
    with open(os.path.join(basedir, 'test_fns_complex.txt'), 'w') as f:
        for fn in test_fns:
            f.write(fn)
            f.write('\n')


def create_train(
        datadir, fns, name, matkey,
        crop_sizes, scales, ksizes, strides, transpose,
        load=h5py.File, augment=True,
        seed=2017, suffix='npz'):
    """
    Create Augmented Dataset
    """

    def preprocess(data):
        data = data.astype(np.float32)
        new_data = []
        data = minmax_normalize(data)
        if transpose is not None:
            data = data.transpose(transpose)
        if datadir.find('ICVL') != -1:
            data = np.rot90(data, k=2, axes=(1, 2))
        # data = minmax_normalize(data.transpose((2,0,1))) # for Remote Sensing
        # Visualize3D(data)
        if crop_sizes is not None:
            data = crop_center(data, crop_sizes[0], crop_sizes[1])

        for i in range(len(scales)):
            if scales[i] != 1:
                temp = zoom(data, zoom=(1, scales[i], scales[i]))
            else:
                temp = data
            temp = Data2Volume(temp, ksizes=ksizes, strides=list(strides[i]))
            new_data.append(temp)
        new_data = np.concatenate(new_data, axis=0)

        if augment:
            for i in range(new_data.shape[0]):
                new_data[i, ...] = data_augmentation(new_data[i, ...])

        return new_data.astype(np.float32)

    np.random.seed(seed)#
    scales = list(scales)
    ksizes = list(ksizes)
    assert len(scales) == len(strides)
    # calculate the shape of dataset
    # try:
    #     data = load(os.path.join(datadir, fns[0]))[matkey]
    #     data = np.array(data)
    # except:
    #     data = loadmat(os.path.join(datadir, fns[0]))[matkey]
    # data = preprocess(data)
    # N = data.shape[0]
    #
    # print('data shape: ', data.shape)
    # map_size = data.nbytes * len(fns) * 1.2
    # print('map size (GB):', map_size / 1024 / 1024 / 1024)
    if suffix == 'npz':
        if os.path.exists(name + '.npz'):
            raise Exception('database already exist!')
        file_dict = {}
        k = 0
        for i, fn in enumerate(fns):
            print(f'processing { i, fn }')
            try:
                X = load(os.path.join(datadir, fn))[matkey]
            except:
                try: X = loadmat(os.path.join(datadir, fn))[matkey]
                except:
                    print('loading', os.path.join(datadir, fn), 'fail')
                    continue
            X = preprocess(X)
            N = X.shape[0]
            for j in range(N):
                str_id = '{:08}'.format(k)
                file_dict[str_id] = X[j]
                k += 1
        print(f'total { k } patch')
        np.savez(name, **file_dict)
    print('done')

def move_data(basedir):
    with open(os.path.join(basedir, 'train_fns.txt'), 'r') as f:
        line = f.read()
        train_fns = line.split('\n')
        train_fns = train_fns[:-1]
    with open(os.path.join(basedir, 'validation_fns_gauss.txt'), 'r') as f:
        line = f.read()
        val_gauss_fns = line.split('\n')
        val_gauss_fns = val_gauss_fns[:-1]
    with open(os.path.join(basedir, 'test_fns_gauss.txt'), 'r') as f:
        line = f.read()
        test_gauss_fns = line.split('\n')
        test_gauss_fns = test_gauss_fns[:-1]
    with open(os.path.join(basedir, 'validation_fns_complex.txt'), 'r') as f:
        line = f.read()
        val_complex_fns = line.split('\n')
        val_complex_fns = val_complex_fns[:-1]
    with open(os.path.join(basedir, 'test_fns_complex.txt'), 'r') as f:
        line = f.read()
        test_complex_fns = line.split('\n')
        test_complex_fns = test_complex_fns[:-1]
    os.makedirs(os.path.join(basedir, 'filefolder', 'train'), exist_ok=True)
    os.makedirs(os.path.join(basedir, 'filefolder', 'val_gauss'), exist_ok=True)
    os.makedirs(os.path.join(basedir, 'filefolder', 'test_gauss'), exist_ok=True)
    os.makedirs(os.path.join(basedir, 'filefolder', 'val_complex'), exist_ok=True)
    os.makedirs(os.path.join(basedir, 'filefolder', 'test_complex'), exist_ok=True)
    for fn in train_fns:
        shutil.copy(os.path.join(basedir, 'filefolder', fn), os.path.join(basedir, 'filefolder', 'train', fn))
    for fn in val_gauss_fns:
        shutil.copy(os.path.join(basedir, 'filefolder', fn), os.path.join(basedir, 'filefolder', 'val_gauss', fn))
    for fn in test_gauss_fns:
        shutil.copy(os.path.join(basedir, 'filefolder', fn), os.path.join(basedir, 'filefolder', 'test_gauss', fn))
    for fn in val_complex_fns:
        shutil.copy(os.path.join(basedir, 'filefolder', fn), os.path.join(basedir, 'filefolder', 'val_complex', fn))
    for fn in test_complex_fns:
        shutil.copy(os.path.join(basedir, 'filefolder', fn), os.path.join(basedir, 'filefolder', 'test_complex', fn))



def create_icvl64_31(datadir, suffix):
    print('create icvl64_31...')
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    savepath = './data/ICVL/trainset'
    os.makedirs(savepath, exist_ok=True)
    savepath = os.path.join(savepath, 'ICVL64_31.npz')
    create_train(
        datadir, fns, savepath, 'rad',  # your own dataset address
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25),
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32)],
        transpose=None,
        load=h5py.File, augment=True, suffix=suffix
    )
def create_icvl512_31(datadir, suffix):

    print('create icvl512_31...')
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    savepath = './data/ICVL/trainset'
    os.makedirs(savepath, exist_ok=True)
    savepath = os.path.join(savepath, 'ICVL512_31.npz')
    create_train(
        datadir, fns, savepath, 'rad',  # your own dataset address
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25),
        ksizes=(31, 512, 512),
        strides=[(31, 512, 512), (31, 256, 256), (31, 256, 256)],
        transpose=None,
        load=h5py.File, augment=True, suffix=suffix
    )

def create_cave64_31(datadir, suffix):
    print('create cave64_31...')
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    savepath = './data/CAVE/trainset'
    os.makedirs(savepath, exist_ok=True)
    savepath = os.path.join(savepath, 'CAVE64_31.npz')
    create_train(
        datadir, fns, savepath, 'imgDouble',  # your own dataset address
        crop_sizes=(512, 512),
        scales=(1, 0.5, 0.25),
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32)],
        transpose=(2, 0, 1),
        load=h5py.File, augment=True, suffix=suffix
    )


if __name__ == '__main__':
    basedir = './data/ICVL/' # your own data address
    # creat_train_val_test(basedir, 0.7, 0.1, 0.2)
    move_data(basedir)
    create_icvl64_31(os.path.join(basedir, 'filefolder', 'train'), suffix = 'npz')





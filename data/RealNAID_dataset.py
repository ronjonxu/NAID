import numpy as np
import os
from data.base_dataset import BaseDataset
from .imlib import imlib
from multiprocessing.dummy import Pool
from tqdm import tqdm
from util.util import augment
from math import log10, pi
import random

#  dataset
class RealNAIDDataset(BaseDataset):
    def __init__(self, opt, split='train', dataset_name='RealNAID'):
        super(RealNAIDDataset, self).__init__(opt, split, dataset_name)

        if self.root == '':
            rootlist = ['/Data/dataset/Real-NAID/']
            for root in rootlist:
                if os.path.isdir(root):
                    self.root = root
                    break

        self.batch_size = opt.batch_size
        self.patch_size = opt.patch_size 
        self.mask_size = opt.mask_size
        self.mask_count = opt.patch_size // opt.mask_size
        self.mode = opt.mode  # RGB, Y or L
        self.imio = imlib(self.mode, lib=opt.imlib)
        self.imio_L = imlib('L', lib=opt.imlib)
        self.names, self.rgb_dirs, self.gt_dirs,self.ir_dirs = self._get_image_dir(self.root, split)
        if split == 'train':
            self._getitem = self._getitem_train
            self.len_data = 1600*10
        elif split == 'val':
            self._getitem = self._getitem_test
            self.len_data =  30*3
        elif split == 'test':
            self._getitem = self._getitem_test
            self.len_data =  30*3
        else:
            raise ValueError

        self.rgb_images = [0] * len(self.names)
        self.ir_images = [0] * len(self.names)
        self.gt_images = [0] * len(self.names)
        read_images(self)           

    def __getitem__(self, index):
        return self._getitem(index)

    def __len__(self):
        return self.len_data

    def _getitem_train(self, idx):
        idx = idx % len(self.names)

        rgb_img = self.rgb_images[idx]
        ir_img = self.ir_images[idx]
        gt_img = self.gt_images[idx]
        rgb_img, ir_img, gt_img = self._crop_patch(rgb_img, ir_img, gt_img)

        rgb_img, ir_img, gt_img = augment(rgb_img, ir_img, gt_img)
        rgb_img = np.float32(rgb_img) / 255.
        ir_img = np.float32(ir_img) / 255.
        gt_img = np.float32(gt_img) / 255.

        return {'noise_img': rgb_img,
                'gt_img': gt_img,
                'ir_img': ir_img,
                'fname': self.names[idx]}

    def _getitem_test(self, idx):

        rgb_img = self.rgb_images[idx]
        ir_img = self.ir_images[idx]
        gt_img = self.gt_images[idx]

        rgb_img = np.float32(rgb_img) / 255.
        ir_img = np.float32(ir_img) / 255.
        gt_img = np.float32(gt_img) / 255.

        return {'noise_img': rgb_img,
                'gt_img': gt_img,
                'ir_img': ir_img,
                'fname': self.names[idx]}


    def _crop_patch(self,  rgb_img, ir_img, gt_img):
        ih, iw = ir_img.shape[-2:]
        p = self.patch_size
        pw = random.randrange(0, iw - p + 1) 
        ph = random.randrange(0, ih - p + 1) 
        return rgb_img[..., ph:ph+p, pw:pw+p], ir_img[..., ph:ph+p, pw:pw+p], \
               gt_img[..., ph:ph+p, pw:pw+p]

    def _get_image_dir(self, dataroot, split=None):
        gt_dirs = []
        rgb_dirs = []
        ir_dirs = []
        image_names = []

        if split == 'train' :
            for image_file in os.listdir(dataroot + 'train/input_rgb/'):
                image_names.append(image_file)
                rgb_dirs.append(dataroot + 'trian/input_rgb/' + image_file)
                ir_dirs.append(dataroot + 'train/input_nir/' + image_file.split('_')[0]+'_nir.png')
                gt_dirs.append(dataroot + 'train/gt/' +image_file.split('_')[0]+'_gt.png')
        elif split == 'val'  or split == 'test':
            for image_file in os.listdir(dataroot + 'test/input_rgb/'):
                image_names.append(image_file)
                rgb_dirs.append(dataroot + 'test/input_rgb/' + image_file)
                ir_dirs.append(dataroot + 'test/input_nir/' + image_file.split('_')[0]+'_nir.png')
                gt_dirs.append(dataroot + 'test/gt/' +image_file.split('_')[0]+'_gt.png')
        else:
            raise ValueError

        image_names = sorted(image_names)
        rgb_dirs = sorted(rgb_dirs)
        gt_dirs = sorted(gt_dirs) 
        ir_dirs = sorted(ir_dirs) 
        return image_names, rgb_dirs, gt_dirs, ir_dirs


def iter_obj(num, objs):
    for i in range(num):
        yield (i, objs)

def imreader(arg):
    i, obj = arg
    for _ in range(3):
        try:
            obj.gt_images[i] = obj.imio.read(obj.gt_dirs[i])
            obj.rgb_images[i] = obj.imio.read(obj.rgb_dirs[i])
            obj.ir_images[i] = obj.imio_L.read(obj.ir_dirs[i])
            failed = False
            break
        except:
            failed = True
    if failed: print(i, '%s fails!' % obj.names[i])

def read_images(obj):
    # may use `from multiprocessing import Pool` instead, but less efficient and
    # NOTE: `multiprocessing.Pool` will duplicate given object for each process.
    print('Starting to load images via multiple imreaders')
    pool = Pool() # use all threads by default
    for _ in tqdm(pool.imap(imreader, iter_obj(len(obj.names), obj)), total=len(obj.names)):
        pass
    pool.close()
    pool.join()

if __name__ == '__main__':
    pass

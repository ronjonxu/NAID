import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
from util.util import calc_psnr as calc_psnr
import time
import numpy as np
from collections import OrderedDict as odict
from copy import deepcopy
import math
import cv2


def calc_psnr_np(sr, hr, range=255.):
	diff = (sr.astype(np.float32) - hr.astype(np.float32)) / range
	mse = np.power(diff, 2).mean()
	return -10 * math.log10(mse)

if __name__ == '__main__':
    opt = TestOptions().parse()

    if not isinstance(opt.load_iter, list):
        load_iters = [opt.load_iter]
    else:
        load_iters = deepcopy(opt.load_iter)

    if not isinstance(opt.dataset_name, list):
        dataset_names = [opt.dataset_name]
    else:
        dataset_names = deepcopy(opt.dataset_name)
    datasets = odict()
    split = 'test'
    for dataset_name in dataset_names:
        dataset = create_dataset(dataset_name, split, opt)
        print(len(dataset))
        datasets[dataset_name] = tqdm(dataset)

    for load_iter in load_iters:
        opt.load_iter = load_iter
        model = create_model(opt)
        model.setup(opt)
        model.eval()

        for dataset_name in dataset_names:
            opt.dataset_name = dataset_name
            tqdm_val = datasets[dataset_name]
            dataset_test = tqdm_val.iterable
            dataset_size_test = len(dataset_test)

            print('='*80)
            print(dataset_name + ' dataset')
            tqdm_val.reset()

            psnr = [0.0] * dataset_size_test
            time_val = 0
            print( dataset_size_test)
            for i, data in enumerate(tqdm_val):
                torch.cuda.empty_cache()
                model.set_input(data, -2)
                torch.cuda.synchronize()
                time_val_start = time.time()
                model.test()
                torch.cuda.synchronize()
                time_val += time.time() - time_val_start
                res = model.get_current_visuals()


                if opt.calc_metrics:
                    psnr[i] = calc_psnr(res['data_gt'], res['data_out'])
                
                if opt.save_imgs:
                    file_name_prefix = data['fname'][0]
                    folder_dir_raw = './ckpt/%s/output' % (opt.name)  
                    os.makedirs(folder_dir_raw, exist_ok=True)
                    save_dir = '%s/%s.png' % (folder_dir_raw, file_name_prefix[:-4])
                    dataset_test.imio.write(np.array(res['data_out'][0].cpu()).astype(np.uint8), save_dir)

            avg_psnr = '%.2f'%np.mean(psnr)
            print('Time: %.3f s AVG Time: %.3f ms PSNR: %s \n' % (time_val, time_val/dataset_size_test*1000, avg_psnr))

    for dataset in datasets:
        datasets[dataset].close()

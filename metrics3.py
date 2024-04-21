import numpy as np
import cv2
import torch
import math
from tqdm import tqdm
import lpips
import glob
from skimage.metrics import structural_similarity as ssim
import argparse
import sys
import os


def calc_psnr_np(sr, hr, range=255.):
    diff = (sr.astype(np.float32) - hr.astype(np.float32)) / range
    mse = np.power(diff, 2).mean()
    return -10 * math.log10(mse)


def lpips_norm(img):
    img = img[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
    img = img / (255. / 2.) - 1
    return torch.Tensor(img).to(device)

def calc_lpips(out, target, loss_fn_alex):
    lpips_out = lpips_norm(out)
    lpips_target = lpips_norm(target)
    LPIPS = loss_fn_alex(lpips_out, lpips_target)
    return LPIPS.detach().cpu().item()

def calc_metrics(out, target, loss_fn_alex):
    psnr = calc_psnr_np(out, target)
    SSIM = ssim(out, target, win_size=11, data_range=255, multichannel=True, gaussian_weights=True, channel_axis=2)
    LPIPS = calc_lpips(out, target, loss_fn_alex)
    return np.array([psnr, SSIM, LPIPS], dtype=float)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Metrics for argparse')
    parser.add_argument('--name', type=str, required=True,
                        help='Name of the folder to save models and logs.')	
    parser.add_argument('--dataroot', type=str, default='/home/xrj/RGB-NIR/dataset/xu/')
    parser.add_argument('--device', default="0")
    args = parser.parse_args()

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    loss_fn_alex_v1 = lpips.LPIPS(net='alex', version='0.1').to(device)

    root = sys.path[0]
    files = [
        root + '/ckpt/' + args.name,
    ]
    


    for file in files:
        print('Start to measure images in %s...' % (file))
        metrics1 = np.zeros([30, 3])
        metrics2 = np.zeros([30, 3])
        metrics3 = np.zeros([30, 3])
        log_dir1 = '%s/log_metrics_lownoise.txt' % (file)
        log_dir2 = '%s/log_metrics_middlenoise.txt' % (file)
        log_dir3 = '%s/log_metrics_highnoise.txt' % (file)
        f1 = open(log_dir1, 'a')
        f2 = open(log_dir2, 'a')
        f3 = open(log_dir3, 'a')
        i = 0

        output_folder = '/output/'
        count_1 = 0
        count_2 = 0
        count_3 = 0
        for image_file in tqdm(sorted(list(os.listdir(file + output_folder)))):
            gt = cv2.imread(args.dataroot +  'test/gt/' + image_file[:-9]+'gt.png')[..., ::-1]
            output = cv2.imread(file + output_folder + image_file)[..., ::-1]
            label = int(image_file[-5])
            
            if label == 1:
                metrics1[count_1, 0:3] = calc_metrics(output, gt, loss_fn_alex_v1)
                f1.write(' File %s  :\t PSNR = %.2f, SSIM = %.4f, LPIPS = %.3f \n' 
                    % (image_file, metrics1[count_1, 0], metrics1[count_1, 1], metrics1[count_1, 2]))
                count_1 += 1
            elif label == 2:
                metrics2[count_2, 0:3] = calc_metrics(output, gt, loss_fn_alex_v1)
                f2.write(' File %s  :\t PSNR = %.2f, SSIM = %.4f, LPIPS = %.3f \n' 
                    % (image_file, metrics2[count_2, 0], metrics2[count_2, 1], metrics2[count_2, 2]))
                count_2 += 1
            else:
                metrics3[count_3, 0:3] = calc_metrics(output, gt, loss_fn_alex_v1)
                f3.write(' File %s  :\t PSNR = %.2f, SSIM = %.4f, LPIPS = %.3f \n' 
                    % (image_file, metrics3[count_3, 0], metrics3[count_3, 1], metrics3[count_3, 2]))
                count_3 += 1

            i = i + 1

        mean_metrics1 = np.mean(metrics1, axis=0)
        mean_metrics2 = np.mean(metrics2, axis=0)
        mean_metrics3 = np.mean(metrics3, axis=0)
        
        print('\n        File        :\t %s \n' % (file))
        print('   Original    GT   :\t PSNR = %.2f, SSIM = %.4f, LPIPS = %.3f \n' 
                % (mean_metrics1[0], mean_metrics1[1], mean_metrics1[2]))

        f1.write('\n        File        :\t %s \n' % (file))
        f1.write('   Original    GT   :\t PSNR = %.2f, SSIM = %.4f, LPIPS = %.3f \n' 
                % (mean_metrics1[0], mean_metrics1[1], mean_metrics1[2]))

        f1.flush()
        f1.close()

        print('\n        File        :\t %s \n' % (file))
        print('   Original    GT   :\t PSNR = %.2f, SSIM = %.4f, LPIPS = %.3f \n' 
                % (mean_metrics2[0], mean_metrics2[1], mean_metrics2[2]))

        f2.write('\n        File        :\t %s \n' % (file))
        f2.write('   Original    GT   :\t PSNR = %.2f, SSIM = %.4f, LPIPS = %.3f \n' 
                % (mean_metrics2[0], mean_metrics2[1], mean_metrics2[2]))

        f2.flush()
        f2.close()

        print('\n        File        :\t %s \n' % (file))
        print('   Original    GT   :\t PSNR = %.2f, SSIM = %.4f, LPIPS = %.3f \n' 
                % (mean_metrics3[0], mean_metrics3[1], mean_metrics3[2]))

        f3.write('\n        File        :\t %s \n' % (file))
        f3.write('   Original    GT   :\t PSNR = %.2f, SSIM = %.4f, LPIPS = %.3f \n' 
                % (mean_metrics3[0], mean_metrics3[1], mean_metrics3[2]))

        f3.flush()
        f3.close()
    
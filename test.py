import os
import torch
import tqdm
import wandb
import numpy as np
from tqdm import tqdm

from argparse import ArgumentParser
from utils.dataset import MTL_TestDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from metrics import SegmentationMetrics, DeblurringMetrics, HomographyMetrics
from models.MIMOUNet.MIMOUNet import VideoMIMOUNet
from utils.transforms import ToTensor, Normalize
from utils.network_utils import model_load, offsets2homo
import cv2, kornia

def save_outputs(args, seq_name, name, output_dict, prv_mask, gt_dict):
    #masks
    cv2.imwrite('{}/{}'.format(os.path.join(args.out, seq_name, 'masks', 'low'), name),
                torch.argmax(output_dict['segment'][0], 1)[0].cpu().numpy() * 255.0)
    cv2.imwrite('{}/{}'.format(os.path.join(args.out, seq_name, 'masks', 'medium'), name),
                torch.argmax(output_dict['segment'][1], 1)[0].cpu().numpy() * 255.0)
    cv2.imwrite('{}/{}'.format(os.path.join(args.out, seq_name, 'masks', 'high'), name),
                torch.argmax(output_dict['segment'][2], 1)[0].cpu().numpy() * 255.0)
    #images
    cv2.imwrite('{}/{}'.format(os.path.join(args.out, seq_name, 'images', 'low'), name),
                output_dict['deblur'][0][0].permute(1,2,0).cpu().numpy()*255.0)
    cv2.imwrite('{}/{}'.format(os.path.join(args.out, seq_name, 'images', 'medium'), name),
                output_dict['deblur'][1][0].permute(1,2,0).cpu().numpy()*255.0)
    cv2.imwrite('{}/{}'.format(os.path.join(args.out, seq_name, 'images', 'high'), name),
                output_dict['deblur'][2][0].permute(1,2,0).cpu().numpy()*255.0)
    #homographies
    homos = [offsets2homo(output_dict['homography'][num]*(2**(2-num)), 80 * (4), 104 * (4)) for num in range(3)]
    gt_homo = gt_dict['homography'][0]


    curr_mask = torch.argmax(output_dict['segment'][2], 1).unsqueeze(0).float()
    prv_mask = torch.argmax(prv_mask, 1)


    warped_curr_scale0 = kornia.warp_perspective(curr_mask, homos[0], (320, 416))
    warped_curr_scale1 = kornia.warp_perspective(curr_mask, homos[1], (320, 416))
    warped_curr_scale2 = kornia.warp_perspective(curr_mask, homos[2], (320, 416))
    warped_curr_gt = kornia.warp_perspective(curr_mask, gt_homo, (320, 416))

    new = torch.ones_like((warped_curr_gt)) * 128
    new[torch.logical_and(warped_curr_scale0 > 0.5, prv_mask > 0.5)] = 255
    new[torch.logical_and(warped_curr_scale0 < 0.5, prv_mask < 0.5)] = 0
    new_tensor = new.repeat(1,3,1,1)
    new_tensor[:, 0][new_tensor[:, 0]==128] = 225
    new_tensor[:, 1][new_tensor[:, 1]==128] = 0
    new_tensor[:, 2][new_tensor[:, 2]==128] = 0
    cv2.imwrite('{}/{}'.format(os.path.join(args.out, seq_name, 'homographies', 'low'), name),
                new_tensor[0].permute(1, 2, 0).cpu().numpy())

    new = torch.ones_like((warped_curr_gt)) * 128
    new[torch.logical_and(warped_curr_scale1 > 0.5, prv_mask > 0.5)] = 255
    new[torch.logical_and(warped_curr_scale1 < 0.5, prv_mask < 0.5)] = 0
    new_tensor = new.repeat(1,3,1,1)
    new_tensor[:, 0][new_tensor[:, 0]==128] = 225
    new_tensor[:, 1][new_tensor[:, 1]==128] = 0
    new_tensor[:, 2][new_tensor[:, 2]==128] = 0
    cv2.imwrite('{}/{}'.format(os.path.join(args.out, seq_name, 'homographies', 'medium'), name),
                new_tensor[0].permute(1, 2, 0).cpu().numpy())

    new = torch.ones_like((warped_curr_gt)) * 128
    new[torch.logical_and(warped_curr_scale2 > 0.5, prv_mask > 0.5)] = 255
    new[torch.logical_and(warped_curr_scale2 < 0.5, prv_mask < 0.5)] = 0
    new_tensor = new.repeat(1,3,1,1)
    new_tensor[:, 0][new_tensor[:, 0]==128] = 225
    new_tensor[:, 1][new_tensor[:, 1]==128] = 0
    new_tensor[:, 2][new_tensor[:, 2]==128] = 0
    cv2.imwrite('{}/{}'.format(os.path.join(args.out, seq_name, 'homographies', 'high'), name),
                new_tensor[0].permute(1, 2, 0).cpu().numpy())

    new = torch.ones_like((warped_curr_gt)) * 128
    new[torch.logical_and(warped_curr_gt > 0.5, prv_mask > 0.5)] = 255
    new[torch.logical_and(warped_curr_gt < 0.5, prv_mask < 0.5)] = 0
    new_tensor = new.repeat(1,3,1,1)
    new_tensor[:, 0][new_tensor[:, 0]==128] = 225
    new_tensor[:, 1][new_tensor[:, 1]==128] = 0
    new_tensor[:, 2][new_tensor[:, 2]==128] = 0

    cv2.imwrite('{}/{}'.format(os.path.join(args.out, seq_name, 'homographies', 'gt'), name),
                new_tensor[0].permute(1, 2, 0).cpu().numpy())

    return homos[0].cpu().numpy(), homos[1].cpu().numpy(), homos[2].cpu().numpy()



def evaluate(args, dataloader, model, metrics_dict):

    tasks = model.module.tasks
    metrics = [k for task in tasks for k in metrics_dict[task].metrics]
    metric_cumltive = {k: [] for k in metrics}
    metrics_hl = {k: [] for k in metrics}
    model.eval()

    l = ['27_4_4_00070.png', '14_5_8_00019.png', '11_12_4_00064.png', '14_15_8_00050.png', '33_19_8_00322.png',
         '33_21_8_00040.png', '20_3_8_00048.png', '33_3_8_00107.png', '33_13_4_00126.png', '7_5_8_00104.png',
         '8_0_4_00103.png',
         '33_13_4_00125.png', '7_4_8_00217.png', '27_2_8_00139.png', '33_6_4_00143.png', '33_3_8_00215.png',
         '15_3_4_00255.png', '8_5_4_00015.png', '20_2_4_00189.png', '20_10_8_00064.png', '14_9_4_00014.png',
         '27_2_8_00138.png', '14_1_8_00088.png',
         '30_1_8_00052.png', '7_7_4_00312.png', '7_0_8_00003.png', '33_12_4_00024.png', '15_5_4_00162.png',
         '33_13_4_00060.png', '8_0_4_00004.png', '30_1_8_00051.png', '20_9_4_00022.png', '27_1_8_00311.png',
         '20_5_8_00164.png', '14_13_8_00200.png',
         '20_14_8_00023.png', '8_7_4_00161.png', '7_6_4_00265.png', '27_0_8_00003.png', '11_3_8_00180.png',
         '8_6_8_00019.png', '27_5_8_00213.png', '11_3_8_00080.png', '33_21_8_00062.png', '14_2_8_00059.png',
         '7_3_8_00034.png', '33_16_4_00161.png',
         '20_14_8_00024.png', '15_11_4_00067.png', '33_21_8_00061.png', '30_3_4_00093.png', '14_6_8_00196.png',
         '15_4_4_00266.png', '33_17_4_00010.png', '33_10_4_00025.png', '30_0_4_00237.png', '14_15_8_00049.png',
         '11_0_4_00003.png', '8_6_8_00111.png',
         '33_11_8_00033.png', '14_9_4_00074.png', '33_18_4_00139.png', '14_11_4_00041.png', '33_20_4_00057.png',
         '14_11_4_00003.png', '15_13_8_00093.png', '11_4_8_00006.png', '15_13_8_00087.png', '15_1_8_00154.png',
         '14_5_8_00020.png', '33_17_4_00040.png', '7_4_8_00001.png', '30_3_4_00088.png', '14_3_8_00093.png',
         '14_10_8_00009.png', '20_6_8_00104.png', '27_4_4_00066.png', '15_7_4_00107.png', '7_6_4_00203.png',
         '15_12_4_00080.png', '30_2_4_00034.png', '11_2_8_00047.png', '11_6_8_00004.png', '33_7_8_00078.png',
         '27_4_4_00069.png', '20_10_8_00129.png', '30_4_8_00147.png', '11_8_4_00041.png', '8_8_4_00062.png',
         '20_8_8_00399.png', '33_12_4_00028.png', '14_1_8_00092.png', '11_0_4_00135.png', '15_4_4_00022.png',
         '20_12_8_00026.png', '15_10_4_00002.png', '15_1_8_00153.png', '14_0_4_00004.png', '20_1_4_00179.png',
         '11_3_8_00081.png', '8_2_8_00090.png', '15_12_4_00055.png', '14_6_8_00195.png', '20_3_8_00030.png',
         '14_16_8_00145.png', '33_18_4_00140.png', '14_13_8_00133.png', '33_0_8_00092.png', '11_13_8_00121.png',
         '11_5_4_00001.png', '7_1_8_00062.png', '30_4_8_00301.png', '20_13_8_00045.png', '20_12_8_00079.png',
         '30_4_8_00300.png', '15_9_8_00333.png', '14_16_8_00035.png', '11_4_8_00007.png', '14_3_8_00030.png',
         '33_16_4_00029.png', '27_0_8_00378.png', '33_18_4_00077.png', '7_1_8_00168.png', '33_7_8_00077.png',
         '8_4_4_00046.png', '20_9_4_00023.png', '14_5_8_00045.png', '11_8_4_00001.png', '7_8_4_00003.png',
         '33_15_4_00009.png', '33_11_8_00063.png', '20_7_4_00009.png', '14_19_4_00114.png', '33_16_4_00162.png',
         '14_0_4_00226.png', '20_3_8_00047.png', '11_12_4_00063.png', '7_6_4_00202.png', '11_8_4_00002.png',
         '11_13_8_00025.png', '7_4_8_00216.png', '8_0_4_00003.png', '33_7_8_00044.png', '20_5_8_00165.png',
         '7_7_4_00313.png', '11_4_8_00054.png', '14_10_8_00008.png', '14_10_8_00084.png', '14_15_8_00101.png',
         '20_1_4_00118.png', '33_9_8_00013.png', '11_7_8_00022.png', '7_2_8_00002.png', '15_0_4_00003.png',
         '11_1_4_00271.png', '14_17_8_00077.png', '33_14_4_00016.png', '8_7_4_00142.png', '27_1_8_00025.png',
         '33_2_8_00031.png', '15_0_4_00004.png', '15_10_4_00003.png', '14_14_8_00065.png', '14_18_8_00005.png',
         '33_10_4_00086.png', '20_5_8_00003.png', '8_10_8_00023.png', '11_11_8_00055.png', '11_14_8_00001.png',
         '15_12_4_00056.png', '8_4_4_00047.png', '7_5_8_00105.png', '20_6_8_00160.png', '15_7_4_00005.png',
         '7_1_8_00063.png', '14_2_8_00107.png', '15_4_4_00265.png', '14_16_8_00034.png', '7_9_4_00222.png',
         '33_16_4_00030.png', '7_0_8_00004.png', '33_17_4_00009.png', '15_5_4_00163.png', '15_1_8_00259.png',
         '33_19_8_00253.png', '8_10_8_00022.png', '14_13_8_00199.png', '33_15_4_00008.png', '27_2_8_00061.png',
         '14_4_8_00202.png', '15_13_8_00094.png', '15_4_4_00021.png', '8_1_4_00054.png', '14_18_8_00069.png',
         '8_7_4_00162.png', '15_9_8_00104.png', '7_9_4_00014.png', '11_1_4_00272.png', '20_10_8_00128.png',
         '20_6_8_00159.png', '14_0_4_00227.png', '11_14_8_00002.png', '14_1_8_00087.png', '20_8_8_00400.png',
         '8_10_8_00144.png', '11_11_8_00044.png', '14_7_4_00142.png', '20_14_8_00001.png', '14_14_8_00069.png',
         '11_2_8_00048.png', '15_12_4_00079.png', '7_6_4_00266.png', '33_2_8_00013.png', '27_3_8_00125.png',
         '20_4_8_00032.png', '7_3_8_00144.png', '11_10_8_00070.png', '11_4_8_00055.png', '11_10_8_00069.png',
         '33_1_4_00031.png', '11_9_8_00069.png', '11_5_4_00017.png', '33_20_4_00074.png', '20_11_4_00012.png',
         '20_7_4_00010.png', '27_4_4_00065.png', '8_3_4_00058.png', '8_4_4_00065.png', '33_20_4_00056.png',
         '30_2_4_00003.png', '20_9_4_00038.png', '11_5_4_00002.png', '20_13_8_00127.png', '20_0_8_00119.png',
         '33_8_8_00073.png', '14_8_4_00073.png', '33_1_4_00015.png', '14_8_4_00072.png', '11_2_8_00029.png',
         '33_8_8_00019.png', '11_0_4_00004.png', '27_3_8_00205.png', '14_19_4_00047.png', '8_3_4_00061.png',
         '33_6_4_00017.png', '27_3_8_00206.png', '33_5_4_00138.png', '7_7_4_00207.png', '20_8_8_00010.png',
         '15_11_4_00005.png', '14_6_8_00003.png', '27_1_8_00026.png', '33_20_4_00073.png', '7_9_4_00015.png',
         '20_14_8_00002.png', '33_9_8_00058.png', '14_15_8_00100.png', '33_1_4_00066.png', '33_3_8_00216.png',
         '11_7_8_00092.png', '7_5_8_00001.png', '11_3_8_00179.png', '20_11_4_00058.png', '8_6_8_00110.png',
         '8_8_4_00134.png', '15_0_4_00084.png', '15_5_4_00013.png', '30_0_4_00003.png', '33_0_8_00003.png',
         '33_9_8_00057.png', '33_2_8_00014.png', '8_9_4_00105.png', '15_2_8_00067.png', '20_4_8_00031.png',
         '8_2_8_00089.png', '20_11_4_00013.png', '11_10_8_00018.png', '20_11_4_00057.png', '15_6_4_00166.png',
         '11_9_8_00002.png', '14_8_4_00002.png', '20_9_4_00037.png', '33_10_4_00026.png', '8_5_4_00059.png',
         '14_12_8_00019.png', '7_2_8_00055.png', '33_13_4_00061.png', '33_17_4_00041.png', '20_10_8_00063.png',
         '20_0_8_00004.png', '14_11_4_00042.png', '14_11_4_00004.png', '11_14_8_00234.png', '33_4_4_00071.png',
         '14_4_8_00203.png', '20_6_8_00103.png', '27_1_8_00310.png', '7_8_4_00204.png', '14_14_8_00070.png',
         '33_6_4_00142.png', '15_8_4_00044.png', '27_5_8_00212.png', '8_3_4_00062.png', '11_2_8_00030.png',
         '11_0_4_00134.png', '27_0_8_00379.png', '7_8_4_00004.png', '15_6_4_00007.png', '14_12_8_00044.png',
         '33_5_4_00126.png', '27_2_8_00062.png', '7_2_8_00003.png', '8_9_4_00106.png', '20_1_4_00117.png',
         '7_9_4_00223.png', '8_4_4_00064.png', '15_2_8_00125.png', '7_0_8_00223.png', '15_8_4_00134.png',
         '14_4_8_00002.png', '11_9_8_00070.png', '30_1_8_00075.png', '20_5_8_00004.png', '15_2_8_00124.png',
         '7_8_4_00205.png', '15_1_8_00260.png', '30_1_8_00074.png', '33_4_4_00054.png', '7_1_8_00169.png',
         '33_9_8_00012.png', '15_9_8_00103.png', '30_3_4_00087.png', '11_1_4_00095.png', '14_17_8_00145.png',
         '14_19_4_00046.png', '15_2_8_00066.png', '8_1_4_00006.png', '8_2_8_00086.png', '8_1_4_00055.png',
         '14_2_8_00060.png', '14_8_4_00001.png', '14_14_8_00066.png', '8_7_4_00141.png', '14_19_4_00113.png',
         '14_12_8_00045.png', '11_5_4_00018.png', '30_0_4_00004.png', '15_5_4_00012.png', '8_5_4_00060.png',
         '33_5_4_00127.png', '11_11_8_00043.png', '20_0_8_00118.png', '11_7_8_00021.png', '11_13_8_00120.png',
         '33_2_8_00030.png', '15_10_4_00034.png', '14_9_4_00075.png', '14_4_8_00001.png', '33_21_8_00039.png',
         '7_3_8_00035.png', '11_6_8_00084.png', '33_0_8_00004.png', '8_5_4_00016.png', '11_11_8_00054.png',
         '8_8_4_00135.png', '15_6_4_00167.png', '33_19_8_00321.png', '27_3_8_00126.png', '33_3_8_00106.png',
         '14_3_8_00029.png', '20_4_8_00089.png', '11_14_8_00235.png', '14_5_8_00044.png', '33_10_4_00085.png',
         '7_5_8_00002.png', '20_7_4_00028.png', '33_4_4_00053.png', '11_9_8_00001.png', '15_7_4_00106.png',
         '14_17_8_00146.png', '30_2_4_00002.png', '14_12_8_00020.png', '30_3_4_00092.png', '15_7_4_00004.png',
         '14_18_8_00006.png', '20_13_8_00126.png', '20_0_8_00003.png', '11_8_4_00040.png', '33_6_4_00018.png',
         '7_0_8_00222.png', '7_2_8_00054.png', '11_12_4_00108.png', '15_3_4_00254.png', '14_18_8_00068.png',
         '15_10_4_00033.png', '30_2_4_00033.png', '8_1_4_00007.png', '33_18_4_00076.png', '14_0_4_00003.png',
         '30_0_4_00238.png', '15_11_4_00066.png', '11_13_8_00024.png', '33_14_4_00025.png', '11_10_8_00017.png',
         '7_3_8_00143.png', '33_7_8_00043.png', '11_12_4_00109.png', '15_9_8_00334.png', '8_6_8_00018.png',
         '14_2_8_00108.png', '20_12_8_00025.png', '20_7_4_00027.png', '20_2_4_00001.png', '15_0_4_00083.png',
         '14_10_8_00085.png', '27_0_8_00004.png', '8_9_4_00038.png', '8_8_4_00063.png', '15_11_4_00004.png',
         '8_10_8_00143.png', '15_13_8_00088.png', '15_8_4_00045.png', '20_3_8_00029.png', '27_5_8_00040.png',
         '14_1_8_00093.png', '20_2_4_00190.png', '8_0_4_00102.png', '33_12_4_00025.png', '33_19_8_00252.png',
         '20_1_4_00178.png', '33_11_8_00062.png', '27_5_8_00041.png', '33_8_8_00018.png', '8_3_4_00059.png',
         '15_6_4_00008.png', '20_4_8_00088.png', '20_13_8_00046.png', '8_9_4_00039.png', '14_7_4_00162.png',
         '11_1_4_00096.png', '8_2_8_00085.png', '11_6_8_00003.png', '33_4_4_00072.png', '14_3_8_00094.png',
         '33_12_4_00027.png', '14_6_8_00004.png', '20_2_4_00002.png', '33_11_8_00032.png', '33_5_4_00137.png',
         '33_1_4_00099.png', '14_13_8_00132.png', '14_16_8_00144.png', '33_15_4_00049.png', '14_17_8_00078.png',
         '11_6_8_00083.png', '14_7_4_00141.png', '20_12_8_00078.png', '14_9_4_00013.png', '7_7_4_00208.png',
         '33_8_8_00072.png', '33_15_4_00050.png', '20_8_8_00009.png', '11_7_8_00091.png', '33_0_8_00091.png',
         '33_14_4_00015.png', '15_3_4_00211.png', '33_14_4_00026.png', '7_4_8_00002.png', '15_8_4_00135.png',
         '15_3_4_00210.png', '14_7_4_00161.png', '30_4_8_00148.png']

    for seq_idx, seq in enumerate(dataloader['val']):

        seq_name = seq['meta']['paths'][0][0].split('/')[-3]
        print(seq_name)
        [os.makedirs(os.path.join(args.out, seq_name, 'masks', scale), exist_ok=True) for scale in ['low', 'medium', 'high']]
        [os.makedirs(os.path.join(args.out, seq_name, 'images', scale), exist_ok=True) for scale in ['low', 'medium', 'high']]
        [os.makedirs(os.path.join(args.out, seq_name, 'homographies', scale), exist_ok=True) for scale in ['low', 'medium', 'high', 'gt']]
        [os.makedirs(os.path.join(args.out, seq_name, 'offsets', scale), exist_ok=True) for scale in ['low', 'medium', 'high']]

        results = open(os.path.join(args.out, seq_name, 'results.txt'), 'w')

        homos_seq_0, homos_seq_1, homos_seq_2 = [], [], []
        for frame in tqdm(range(args.prev_frames, len(seq['meta']['paths']))):

            path = [x.split('/')[-3] + '_' + x.split('/')[-1] for x in seq['meta']['paths'][frame]]
            # Load the data and mount them on cuda
            if frame == args.prev_frames:
                frames = [seq['image'][i].cuda(non_blocking=True) for i in range(frame + 1)]
                m2 = [torch.zeros((frames[0].shape[0], 2, 80, 104), device='cuda'),
                      torch.zeros((frames[0].shape[0], 2, 160, 208), device='cuda'),
                      torch.zeros((frames[0].shape[0], 2, 320, 416),device='cuda')]
            else:
                frames.append(seq['image'][frame].cuda(non_blocking=True))
                frames.pop(0)

            gt_dict = {task: seq[task][frame].cuda(non_blocking=True) if type(seq[task][frame]) is torch.Tensor else
            [e.cuda(non_blocking=True) for e in seq[task][frame]] for task in tasks}

            with torch.no_grad():
                outputs = model(frames[0], frames[1], m2)
            output_dict = dict(zip(tasks, outputs))

            name = seq['meta']['paths'][frame][0].split('/')[-1]
            homo0, homo1, homo2 = save_outputs(args, seq_name, name, output_dict, m2[2], gt_dict)
            homos_seq_0.append(homo0)
            homos_seq_1.append(homo1)
            homos_seq_2.append(homo2)

            m2 = output_dict['segment']
            task_metrics = {task: metrics_dict[task](output_dict[task], gt_dict[task]) for task in tasks}
            metrics_values = {k: torch.round((10**3 * v))/(10**3) for task in tasks for k, v in task_metrics[task].items()}


            for metric in metrics:
                metric_cumltive[metric].append(metrics_values[metric])
                if path[0] in l:
                    metrics_hl[metric].append(metrics_values[metric])

        np.save( os.path.join(args.out, seq_name, 'homos0.npy'), np.concatenate(homos_seq_0, 0))
        np.save(os.path.join(args.out, seq_name, 'homos1.npy'), np.concatenate(homos_seq_1, 0))
        np.save(os.path.join(args.out, seq_name, 'homos2.npy'), np.concatenate(homos_seq_2, 0))

        results.close()
    metric_averages = {m: sum(metric_cumltive[m])/len(metric_cumltive[m]) for m in metrics}
    metric_hl_averages = {m: sum(metrics_hl[m]) / len(metrics_hl[m]) for m in metrics}

    print("\n[TEST] {}\n".format(' '.join(['{}: {:.3f}'.format(m, metric_averages[m]) for m in metrics])))
    print("\n[TEST-HL] {}\n".format(' '.join(['{}: {:.3f}'.format(m, metric_hl_averages[m]) for m in metrics])))

    wandb_logs = {"Test - {}".format(m): metric_averages[m] for m in metrics}
    wandb_hl_logs = {"Test - HL - {}".format(m): metric_hl_averages[m] for m in metrics}
    wandb.log(wandb_logs)
    wandb.log(wandb_hl_logs)




def main(args):


    tasks = [task for task in ['segment', 'deblur', 'homography'] if getattr(args, task)]

    transformations = {'val': transforms.Compose([ToTensor(), Normalize()])}
    data = {'val': MTL_TestDataset(tasks, args.data_path, 'val', args.seq_len, transform=transformations['val'])}
    loader = {'val': DataLoader(data['val'], batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=False)}


    metrics_dict = {
        'segment': SegmentationMetrics().cuda(),
        'deblur': DeblurringMetrics().cuda(),
        'homography': HomographyMetrics().cuda()}
    metrics_dict = {k: v for k, v in metrics_dict.items() if k in tasks}

    #params, fps, gflops = measure_efficiency()
    #print(params, fps, gflops)

    model = VideoMIMOUNet(tasks).cuda()
    model = torch.nn.DataParallel(model).cuda()

    load_model_path = '/home/efklidis/mtl-dental/models/res-fft/ckpt_82.pth'
    _ = model_load(load_model_path, model)
    os.makedirs(os.path.join(args.out), exist_ok=True)

    wandb.init(project='mtl-normal', entity='dst-cv')#, mode='disabled'
    wandb.run.name = args.out.split('/')[-1]
    wandb.watch(model)

    evaluate(args, loader, model, metrics_dict)


if __name__ == '__main__':
    parser = ArgumentParser(description='Parser of Training Arguments')

    parser.add_argument('--data', dest='data_path', help='Set dataset root_path', default='/media/efklidis/4TB/data_ours_new_split', type=str) #/media/efklidis/4TB/ # ../raid/data_ours_new_split
    parser.add_argument('--out', dest='out', help='Set output path', default='/media/efklidis/4TB/test_results|fft-res|best/', type=str)

    parser.add_argument("--segment", action='store_false', help="Flag for segmentation")
    parser.add_argument("--deblur", action='store_false', help="Flag for  deblurring")
    parser.add_argument("--homography", action='store_false', help="Flag for  homography estimation")
    parser.add_argument("--resume", action='store_false', help="Flag for resume training")

    parser.add_argument('--bs', help='Set size of the batch size', default=1, type=int)
    parser.add_argument('--seq_len', dest='seq_len', help='Set length of the sequence', default=None, type=int)
    parser.add_argument('--prev_frames', dest='prev_frames', help='Set number of previous frames', default=1, type=int)

    parser.add_argument('--save_every', help='Save model every n epochs', default=1, type=int)


    args = parser.parse_args()

    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(args)

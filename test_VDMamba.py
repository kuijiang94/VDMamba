import torch.nn.functional as F
import numpy as np
import os
import argparse
from tqdm import tqdm
import time

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils
from thop import profile

from data_RGB import get_test_data
from VDMamba import WaveMAMBA#FMRNet#COformer

from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Deraining using MSGN')

# parser.add_argument('--input_dir', default='/home/jk/code/deraining/FMRNet/Datasets/dehazing/test/', type=str, help='Directory of validation images')# LLE, underwater,dehazing 
parser.add_argument('--input_dir', default='dataset/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/VDMamba/results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/models/VDMamba/model_latest.pth', type=str, help='Path to weights')#FMRNet_neg, FMRNet_v2,model_best,model_epoch_875, model_latest
parser.add_argument('--gpus', default='2', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus #args.gpus

model_restoration = WaveMAMBA() 

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)

file = 'par_gflops.txt'

model_restoration.cuda()
# model_restoration = nn.DataParallel(model_restoration) 
model_restoration.eval()



datasets = ['3DTest']#['Futaba', 'Hinoki','Koharu', 'Midori', 'Natsume', 'Shirohana', 'Tsubaki']#, 'TESTSET']


for dataset in datasets:
    rgb_dir_test = os.path.join(args.input_dir, dataset, 'input')
    #print("rgb_dir_test",rgb_dir_test)
    if (rgb_dir_test):
        print("rgb_dir_test",rgb_dir_test)
        print('***************************')
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)


    result_dir  = os.path.join(args.result_dir, dataset, 'VDMamba_latest')
    if not os.path.exists(result_dir):
        utils.mkdir(result_dir)
    all_time =0
    count = 0
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            
            input_    = data_test[0].cuda()
            _, _, h, w = input_.shape
            factor = 16
            pad_h = (factor - h % factor) % factor
            pad_w = (factor - w % factor) % factor
            input_padded = F.pad(input_, (0, pad_w, 0, pad_h), mode='reflect')
            
            filenames = data_test[1]
            st_time=time.time()
            restored_padded = model_restoration(input_padded)
            ed_time=time.time()
            cost_time=ed_time-st_time
            all_time +=cost_time
            count +=1
            restored = restored_padded[:, :, :h, :w]
            #print('spent {} s.'.format(cost_time))
            #print(filenames)
            restored = torch.clamp(restored,0,1)
            # restored = torch.clamp(restored[0],0,1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)
    print('spent {} s.'.format(all_time))
    print('spent {} s per item.'.format(all_time/(count)))#(ii+1)
input_test    = input_.cuda()
# print('model_restoration', model_restoration.device)
flops, params = profile(model_restoration, (input_,))
print('flops: ', flops, 'params: ', params)

format_str = ('flops: %.4f, params: %.4f, per_time: %.4f')
a = str(format_str % (flops, params, all_time/(count)))
PSNR_file = open(file, 'a+')
PSNR_file.write(a)
PSNR_file.write('\n')
PSNR_file.close()
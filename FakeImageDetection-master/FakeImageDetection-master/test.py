import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score
import numpy as np
from PIL import Image
import os
import clip
from tqdm import tqdm
import timm
import argparse
import random

import torchvision.models as vis_models

from dataset import *
from augment import ImageAugmentor
from mask import *
from utils import *
import sys
# 导入初始化后的universal模型
# sys.path.append('/root/flask/')
# from ProcessWorker import universal_init_model,universal_model_device

os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['MASTER_ADDR']='127.0.0.1'
os.environ['MASTER_PORT']='1234'
os.environ['RANK'] = '0'  # 当前进程的等级（rank）
os.environ['WORLD_SIZE'] = '1'  # 参与计算的进程总数

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings for your script")

    parser.add_argument(
        '--model_name',
        default='RN50_mod',
        type=str,
        choices=[
            'RN18', 'RN34', 'RN50', 'RN50_mod', 'clip',
        ],
        help='Type of model to use; includes ResNet variants'
        )
    parser.add_argument(
        '--mask_type', 
        default='spectral', 
        choices=[
            'patch', 
            'spectral',
            'pixel', 
            'nomask'], 
        help='Type of mask generator'
        )
    parser.add_argument(
        '--band', 
        default='all',
        type=str,
        choices=[
            'all', 'low', 'mid', 'high',]
        )
    parser.add_argument(
        '--pretrained', 
        action='store_true', 
        help='For pretraining'
        )
    parser.add_argument(
        '--ratio', 
        type=int, 
        default=15,
        help='Ratio of mask to apply'
        )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=1, 
        help='Batch Size'
        )
    parser.add_argument(
        '--data_type', 
        default="Wang_CVPR20", 
        type=str, 
        choices=['Wang_CVPR20', 'Ojha_CVPR23'], 
        help="Dataset Type"
        )
    parser.add_argument(
        '--other_model', 
        action='store_true', 
        help='if the model is from my own code'
        )
    parser.add_argument(
        '--local_rank', 
        default=0, 
        type=int, 
        help='Local rank for distributed training'
        )

    args = parser.parse_args()

    device = torch.device("cuda:0")
    
    # 初始化随机种子
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 初始化分布式进程组
    dist.init_process_group(backend='nccl', init_method='env://')
    
    model_name = args.model_name.lower()
    finetune = 'ft' if args.pretrained else ''
    band = '' if args.band == 'all' else args.band

    if args.mask_type != 'nomask':
        ratio = args.ratio
        checkpoint_path = f'checkpoints/mask_{ratio}/{model_name}{finetune}_{band}{args.mask_type}mask.pth'
    else:
        ratio = 0
        checkpoint_path = f'checkpoints/mask_{ratio}/{model_name}{finetune}.pth'
    checkpoint_path='/root/FakeImageDetection-master/FakeImageDetection-master/checkpoints/mask_15/rn50_modft_spectralmask.pth'
    # Define the path to the results file
    results_path = f'results/{args.data_type.lower()}'
    os.makedirs(results_path, exist_ok=True)
    filename = f'{model_name}{finetune}_{band}{args.mask_type}mask{ratio}.txt'

    # Pretty print the arguments
    print("\nSelected Configuration:")
    print("-" * 30)
    print(f"Dataset Type: {args.data_type}")
    print(f"Model type: {args.model_name}")
    print(f"Ratio of mask: {ratio}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Mask Type: {args.mask_type}")
    print(f"Checkpoint Type: {checkpoint_path}")
    print(f"Results saved to: {results_path}/{filename}")
    print("-" * 30, "\n")

    if args.data_type == 'Wang_CVPR20':
        datasets = {
            'BigGAN': '/root/FakeImageDetection-master/FakeImageDetection-master/testImages',
        }
    # elif args.data_type == 'GenImage':
    #     datasets = {
    #         'VQDM': '/home/users/chandler_doloriel/scratch/Datasets/GenImage/imagenet_vqdm/imagenet_vqdm/val',
    #         'Glide': '/home/users/chandler_doloriel/scratch/Datasets/GenImage/imagenet_glide/imagenet_glide/val',
    #     }
    elif args.data_type == 'Ojha_CVPR23':
        datasets = {
            'Guided': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/guided',
            'LDM_200': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/ldm_200',
            'LDM_200_cfg': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/ldm_200_cfg',
            'LDM_100': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/ldm_100',
            'Glide_100_27': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/glide_100_27',
            'Glide_50_27': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/glide_50_27',
            'Glide_100_10': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/glide_100_10',
            'DALL-E': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/dalle',       
        }
    else:
        raise ValueError("wrong dataset type")

    # Initialize a counter
    dataset_count = len(datasets)

    for dataset_name, dataset_path in datasets.items():
        if args.local_rank == 0:
            print(f"\nEvaluating {dataset_name}")
            
        # 初始化model
        # sys.path.append('/root/flask/')
        # from ProcessWorker import universal_init_model,universal_model_device
        # init_model=universal_init_model
        # if init_model is None:
        #     print("universal模型未被正确初始化！")

        #device=universal_model_device

        avg_ap, avg_acc, auc= evaluate_model(
            #init_model,
            args.model_name,
            args.data_type,
            args.mask_type,
            ratio/100,
            dataset_path,
            args.batch_size,
            checkpoint_path,
            device,
            args,
        )
        if args.local_rank == 0:
            # Write the results to the file
            with open(f'{results_path}/{filename}', 'a') as file:
                if file.tell() == 0: # Check if the file is empty
                    file.write("Selected Configuration:\n")
                    file.write("-" * 28 + "\n")
                    file.write(f"Device: {args.local_rank}\n")
                    file.write(f"Dataset Type: {args.data_type}\n")
                    file.write(f"Model type: {args.model_name}\n")
                    file.write(f"Ratio of mask: {ratio}\n")
                    file.write(f"Batch Size: {args.batch_size}\n")
                    file.write(f"Mask Type: {args.mask_type}\n")
                    file.write(f"Checkpoint Type: {checkpoint_path}\n")
                    file.write(f"Results saved to: {results_path}/{filename}\n")
                    file.write("-" * 28 + "\n\n")
                    file.write("Dataset, Precision, Accuracy, AUC\n")
                    file.write("-" * 28)
                    file.write("\n")
                file.write(f"{dataset_name}, {avg_ap*100:.2f}, {avg_acc*100:.2f}, {auc:.3f}\n")
                #file.write(f"{dataset_name}, {avg_ap*100:.2f}, {avg_acc*100:.2f}")
            # Decrement the counter
            dataset_count -= 1
            if dataset_count == 0:
                with open(f'{results_path}/{filename}', 'a') as file:
                    file.write("-" * 28 + "\n")
                    file.write("\n")

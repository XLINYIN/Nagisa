import os
import argparse
import torch
from PIL import Image

from solver import Solver
from Dataloader import get_loader
from torch.backends import cudnn

def main(config):
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"可用的 CUDA 设备数量: {num_devices}")
        for i in range(num_devices):
            device_name = torch.cuda.get_device_name(i)
            print(f"设备 {i}: {device_name}")

    else:
        print("CUDA 不可用")

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    celeba_loader = get_loader(config.Sample_celeba_image_dir, config.Sample_attr_path, config.selected_attrs,
                               config.batch_size,config.mode)

    CelebASample_loader=get_loader(config.Sample_celeba_image_dir, config.Sample_attr_path, config.selected_attrs,
                               config.batch_size,'Sample')

    solver=Solver(celeba_loader,CelebASample_loader,config)
    if config.mode=='train':
        solver.train()
    else:
        solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #model configuratioin
    Start=input("Start iters")
    Start=int(Start)
    parser.add_argument('--start_iter',type=int,default=Start)
    parser.add_argument('--G_para',type=str)
    parser.add_argument('--D_para', type=str)
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--encode_layers',type=int,default=5,help='number of layers in encode')
    parser.add_argument('--decode_layers', type=int, default=5, help='number of layers in decode')
    parser.add_argument('--discriminator_layers',type=int,default=5,help='number of layers in discriminator')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

    #Training configuration
    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=2000000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--k1',type=float,help='伪图真实性')
    parser.add_argument('--k2',type=float,help='重建真实性')
    parser.add_argument('--k3',type=float,help='原地重建')
    parser.add_argument('--k4',type=float,help='复原重建')
    parser.add_argument('--k5', type=float,help='伪图属性')
    parser.add_argument('--k6', type=float,help='恢复属性')
    parser.add_argument('--k7', type=float,help='伪图L1范数')
    parser.add_argument('--k8', type=float,help='恢复L1范数')
    parser.add_argument('--k9',type=float,help='伪图识别损失')
    parser.add_argument('--k10',type=float,help='恢复识别损失')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Male','Young'])

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--Sample_celeba_image_dir', type=str,
                        default='Dataset/CelebA-HQ128')
    parser.add_argument('--Sample_attr_path', type=str,
                        default='Dataset/CelebA-HQ-attribute-anno.txt')

    parser.add_argument('--log_dir', type=str, default='Nagisa/logs')

    parser.add_argument('--model_save_dir', type=str, default='Nagisa/models')

    parser.add_argument('--sample_dir', type=str, default='Nagisa/simples')
    parser.add_argument('--result_dir', type=str, default='Nagisa/Presults')
    parser.add_argument('--resume_iters',type=int,default=0)
    # Step size.
    parser.add_argument('--log_step', type=int, default=10000)
    parser.add_argument('--sample_step', type=int, default=2000)
    parser.add_argument('--model_save_step', type=int, default=2000)
    parser.add_argument('--lr_update_step', type=int, default=1000)


    config = parser.parse_args()
    # print(config)
    main(config)


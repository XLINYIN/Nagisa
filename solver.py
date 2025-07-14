import model
from model import G_MASK
from model import Discriminators
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
#from torch.utils.tensorboard import SummaryWriter
import shutil
import choose_device
import random
from facenet_pytorch import InceptionResnetV1
def Label_G(label_org):
    label_trg=label_org.clone()
    for i in range(0,label_trg.shape[0]):
        for j in range(0,label_trg.shape[1]):
            random_integer = random.randint(1, 10)
            label_trg[i][j]=1-label_trg[i][j] if random_integer%2==0 else label_trg[i][j]
    return torch.FloatTensor(label_trg)
class Solver(object):
    torch.cuda.set_device(choose_device.Choose_ID)
    def __init__(self,celeba_loader,CelebASample,config):
        self.start_iter=config.start_iter
        self.G_para=config.G_para
        self.D_para=config.D_para
        self.CelebASample=CelebASample
        self.celeba_loader=celeba_loader
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.encode_layers = config.encode_layers
        self.decode_layers = config.decode_layers
        self.discriminator_layers=config.discriminator_layers
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.Sigmoid = nn.Sigmoid()

        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.D_now_rate=-2

        self.k1=config.k1
        self.k2=config.k2
        self.k3=config.k3
        self.k4=config.k4
        self.k5=config.k5
        self.k6=config.k6
        self.k7=config.k7
        self.k8=config.k8
        self.k9 = config.k9
        self.k10 = config.k10

        # 如果存在日志目录并且从0开始迭代，就清空

        self.selected_attrs = config.selected_attrs
        self.device=torch.device(choose_device.Choose_Device)
        torch.cuda.set_device(choose_device.Choose_ID)

        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        self.resume_iters=config.resume_iters
        self.count=0
        self.l1_norm = nn.L1Loss()

        print(self.device)

        self.build_model()

    def build_model(self):

        self.G=model.G_MASK(self.g_conv_dim,self.d_conv_dim,self.encode_layers,self.decode_layers,len(self.selected_attrs),2,self.image_size)
        self.D=model.Discriminators()

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, (self.beta1, self.beta2))
        self.d_optimizer=  torch.optim.Adam(self.D.parameters(), self.d_lr, (self.beta1, self.beta2))
        self.G.to(self.device)
        self.D.to(self.device)

        if(self.start_iter!=0):
            self.D.load_state_dict(torch.load(self.D_para))
            self.G.load_state_dict(torch.load(self.G_para))

        self.Id = InceptionResnetV1(pretrained='vggface2').eval()#身份识别器
        self.Id.classify = False
        self.Id= self.Id.to(self.device)

    def print_network(self, model, name):  # 打印网络
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
            print(p.numel())
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def cosine_similarity(self,embedding1, embedding2):
        dot_product = torch.sum(embedding1*embedding2,dim=1)
        norm1 = torch.norm(embedding1,dim=1)
        norm2 = torch.norm(embedding2,dim=1)
        similarity = dot_product / (norm1 * norm2)
        return torch.mean(similarity)
    def train(self):

        data_loader=self.celeba_loader
        Sample_dataloader=self.CelebASample
        data_iter=iter(data_loader)
        sample_iter=iter(Sample_dataloader)
        Sample_org,SampleLabel_org=next(sample_iter)
        Sample_org = Sample_org.to(self.device)
        SampleLabel_org = SampleLabel_org.to(self.device)
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training.
        start_iters = self.start_iter

        print('Start training...')

        t1=time.time()

        #计算当前学习率
        for i in range(start_iters):
            if (i + 1) % 1000 == 0 and (i + 1) <= 1000000:
                g_lr = 0.996 * g_lr
                d_lr = 0.996 * d_lr
        self.update_lr(g_lr, d_lr)
        print('已经重置{}次学习率， g_lr: {}, d_lr: {}.'.format(start_iters, g_lr, d_lr))

        for i in range(start_iters,self.num_iters):
            try:
                x_real,label_org=next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)
            #每个批次的图片和标签
            label_trg=Label_G(label_org)

            c_org=label_org.clone()
            c_trg=label_trg.clone()

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            if(1==0):
                pass
            else:
                out_src1,out_cls1=self.D(x_real)
                d_loss_real = self.D_now_rate * torch.mean(out_src1)
                d_loss_cls = self.classification_loss(out_cls1, label_org)


                x_fake = self.G(x_real, trg=c_trg,mode='enc-dec')
                x_style=self.G(x_real,mode='ext')
                x_rec=self.G(x_fake,style=x_style,mode='rec')

                out_src2,_=self.D(x_fake.detach())
                out_src3,_=self.D(x_rec.detach())
                d_loss_fake = torch.mean(out_src2)+torch.mean(out_src3)

                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src ,_= self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                d_loss = d_loss_real + d_loss_fake + 24 * d_loss_cls + self.lambda_gp * d_loss_gp
                if (i+1) % 200 == 0:
                    print(f"Iteration {i+1} D_loss:{d_loss.item()} D_cls{d_loss_cls.item()}")
                    print(f"loss_real{d_loss_real.item()} loss_fake{d_loss_fake.item()}")
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # ====================================================================================#
            if(i+1)%2==0:
                #不需要区分考虑重建，需要同时进行训练
                c_org=label_org.clone()
                c_trg=label_trg.clone()
                #self-generation
                x_style=self.G(x_real,mode='ext')

                x_rec1=self.G(x_real,trg=c_trg,mode='enc-dec')

                x_rec2=self.G(x_real,style=x_style,mode='rec') #恢复器原地重建

                x_fake=self.G(x_real,trg=c_trg,mode='enc-dec') #伪图生成


                x_rec3=self.G(x_fake,style=x_style,mode='rec') #恢复生成

                discriminator_output_fakeP,sensitive_attributeP= self.D(x_fake)

                discriminator_output_fakeR,sensitive_attributeR= self.D(x_rec3)

                loss_adv_GP = - torch.mean(discriminator_output_fakeP)
                loss_adv_GR = - torch.mean(discriminator_output_fakeR)

                loss_attr_GP = self.classification_loss(sensitive_attributeP, c_trg)
                loss_attr_GR = self.classification_loss(sensitive_attributeR, c_org)

                # information loss 两个目标图的信息损失
                g_loss_fake_L1 = self.l1_norm(x_fake, x_real)
                g_loss_reco_L1 = self.l1_norm(x_rec3,x_real)

                # reconstruction loss 两个重建图的重建损失
                g_loss_rec1=torch.mean(torch.abs(x_rec1-x_real))
                g_loss_rec2 = torch.mean(torch.abs(x_rec2 - x_real))

                #身份识别
                real1=self.Id(x_real)
                fake1=self.Id(x_fake)
                fake2=self.Id(x_rec3)
                distance1 = torch.norm(real1 - fake1, p=2)
                distance2 = torch.norm(real1 - fake2, p=2)
                Similarity=[self.cosine_similarity(real1,fake1),self.cosine_similarity(real1,fake2)]
                g_loss_id=self.k9*distance1+self.k10*distance2



                loss_total_G=self.k1*loss_adv_GP+self.k3*g_loss_rec1+self.k5*loss_attr_GP+self.k7*g_loss_fake_L1\
                            +self.k2*loss_adv_GR+self.k4*g_loss_rec2+self.k6*loss_attr_GR+self.k8*g_loss_reco_L1+g_loss_id


                if (i + 1) % 200 == 0:
                    print(f"Iteration{i+1} - LossG: {loss_total_G.item()} "
                          f"Fake(P,R){str(self.k1*loss_adv_GP.item()),str(self.k2*loss_adv_GR.item())}"
                          f"Rec(1,2){self.k3*g_loss_rec1.item(),self.k4*g_loss_rec2.item()}"
                          f"Att(P,R){self.k5*loss_attr_GP.item(),self.k6*loss_attr_GR.item()}"
                          f"L1(1,2){self.k7*g_loss_fake_L1.item(),self.k8*g_loss_reco_L1.item()}"
                          f"ID(1,2){self.k9*distance1.item(),self.k10*distance2.item()}"
                          f"Similarity(1,2){Similarity[0].item(),Similarity[1].item()}")
                    t2 = time.time()
                    print(f"MAP：轮数：{i + 1}, 耗时：", int(t2 - t1) // 3600, "h", (int(t2 - t1) % 3600) // 60, "m",(int(t2 - t1) % 3600) % 60, "s")
                    t1 = time.time()

                self.reset_grad()
                loss_total_G.backward()
                self.g_optimizer.step()







            if (i + 1) % 2000 == 0:
                print("saving......")
                with torch.no_grad():

                    x_fake_list = [Sample_org]
                    self.myfix_PaM_train=[[1,0],[1,0],[1,0],[1,0],[0,0],[0,1],[0,0],[0,1]]
                    x_fixed_list = torch.Tensor(self.myfix_PaM_train)
                    x_fixed_list = x_fixed_list.to(self.device)
                    x_style=self.G(Sample_org,mode='ext')
                    x_fake=self.G(Sample_org, x_fixed_list,mode='enc-dec')
                    x_rec=self.G(x_fake,style=x_style,mode='rec')
                    x_fake_list.append(x_fake)
                    x_fake_list.append(x_rec)
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))




            if (i + 1) % 1000 == 0 and (i + 1) <= 1000000:
                g_lr = 0.996 * g_lr  # 更新学习率,99.6%
                d_lr = 0.996 * d_lr
                self.update_lr(g_lr, d_lr)

            if (i + 1) % 10000 == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path=os.path.join(self.model_save_dir,'{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(),D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))





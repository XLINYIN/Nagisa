import torch.nn as nn
import torch
import math
import numpy as np

def add_normalization_2d(layers,norm_fn,n_out):
    if norm_fn=='Batchnorm':
        layers.append(nn.BatchNorm2d(n_out))
    elif norm_fn=='Instancenorm':
        layers.append(nn.InstanceNorm2d(n_out))
    elif norm_fn=='none':
        pass

def add_activation(layers,fn,leakyrelu=0.02):
    if fn=='none':
        pass
    elif fn == 'relu':
        layers.append(nn.ReLU())
    elif fn == 'lrelu':
        layers.append(nn.LeakyReLU(leakyrelu))
    elif fn == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif fn == 'tanh':
        layers.append(nn.Tanh())

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x=x + self.conv_block(x)
        return x/math.sqrt(2)

class Conv2dBlock(nn.Module):
    def __init__(self, dim_in, dim_out, activ=nn.LeakyReLU(0.02),Norm='Batch'):
        super().__init__()
        self.activ = activ
        self._build_weights(dim_in, dim_out,Norm)
    def _build_weights(self, dim_in, dim_out,Norm='Batch'):
        self.conv1 = nn.Conv2d(dim_out, dim_out, 4, 2, 1)
        self.conv2=nn.Conv2d(dim_in,dim_out,3,1,1)
        if Norm=='Batch':
            self.Nom1=nn.BatchNorm2d(dim_out)
            self.Nom2=nn.BatchNorm2d(dim_out)
        else:
            self.Nom1=nn.InstanceNorm2d(dim_out)
            self.Nom2=nn.InstanceNorm2d(dim_out)
    def forward(self, x):
        x=self.conv2(x)
        x=self.Nom2(x)
        x=self.activ(x)
        x = self.conv1(x)
        x = self.Nom1(x)
        x = self.activ(x)
        return x



class Conv2dBlockWithRes(nn.Module):
    def __init__(self, dim_in, dim_out, activ=nn.LeakyReLU(0.02), Norm='Batch'):
        super().__init__()
        self.activ = activ
        self._build_weights(dim_in, dim_out, Norm)
    def _build_weights(self, dim_in, dim_out, Norm='Batch'):
        self.conv1 = nn.Conv2d(dim_out, dim_out, 4, 2, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv3 = nn.Conv2d(dim_in,dim_out,1,1,0)
        self.conv4 = nn.Conv2d(dim_out,dim_out,3,1,1)
        if Norm == 'Batch':
            self.Nom1 = nn.BatchNorm2d(dim_out)
            self.Nom2 = nn.BatchNorm2d(dim_out)
            self.Nom3 = nn.BatchNorm2d(dim_out)
        else:
            self.Nom1 = nn.InstanceNorm2d(dim_out)
            self.Nom2 = nn.InstanceNorm2d(dim_out)
            self.Nom3 = nn.InstanceNorm2d(dim_out)

    def forward(self, x):
        residual=self.conv3(x)
        x = self.conv2(x)
        x = self.Nom2(x)
        x = self.activ(x)
        x = self.conv4(x)
        x = self.Nom3(x)
        x = self.activ(x)
        x = (x+residual)/math.sqrt(2)

        x = self.conv1(x)
        x = self.Nom1(x)
        x = self.activ(x)
        return x


class ConvTranspose2dBlock(nn.Module):
    def __init__(self,n_in,n_out,kernel_size,stride,padding,norm_fn,activ_fn):
        super(ConvTranspose2dBlock,self).__init__()
        layers=[ResidualBlock(n_in),nn.ConvTranspose2d(n_in,n_out,kernel_size,stride=stride,padding=padding)]
        add_normalization_2d(layers,norm_fn,n_out)
        add_activation(layers,activ_fn)
        self.layers=nn.Sequential(*layers)#*用于解包
    def forward(self,x):
        return self.layers(x)


class G_MASK(nn.Module):
    def __init__(self,encode_dim=64,decode_dim=64,encode_layers=5,decode_layers=5,n_attrs=2,shortcut_layers=2,img_size=128,extract_dim=16,extract_layers=5,style_dim=256):
        super().__init__()


        self.final_size=img_size//2**encode_layers  #128*128->4*4
        n_in=3
        self.shortcut_layers=shortcut_layers
        self.encode=[]
        for i in range(0,encode_layers):
            n_out=encode_dim*(2**i)
            self.encode.append(Conv2dBlock(n_in,n_out,Norm='Batch'))
            n_in=n_out
        #3->64,64->128,....512->1024
        self.encode=nn.ModuleList(self.encode)



        ##Decoder
        layers=[]
        n_in=1024+n_attrs#1026
        for i in range(0,decode_layers):
            if i <decode_layers-1:
                n_out=decode_dim*2**(decode_layers-i-1)# 1024,512,256,128,64
                layers.append(ConvTranspose2dBlock(n_in,n_out,4,2,1,norm_fn='Batchnorm',activ_fn='relu'))
                print(n_in,n_out,i)
                n_in=n_out
                n_in=n_in+n_in//2 if self.shortcut_layers > i else n_in
            else:
                layers+=[ConvTranspose2dBlock(n_in,3,4,stride=2,padding=1,norm_fn='Batchnorm',activ_fn='tanh')]
        self.dec_layers=nn.ModuleList(layers)

        ##ReConstructurer
        layers2=[]
        n_in=1024+style_dim
        for i in range(0,decode_layers):#2048->1024
            if i <decode_layers-1:
                n_out=decode_dim*2**(decode_layers-i-1)# 1024,512,256,128,64
                layers2.append(ConvTranspose2dBlock(n_in,n_out,4,2,1,norm_fn='Batchnorm',activ_fn='relu'))
                print(n_in,n_out,i)
                n_in=n_out
                n_in=n_in+n_in//2 if self.shortcut_layers > i else n_in
            else:
                layers2+=[ConvTranspose2dBlock(n_in,3,4,stride=2,padding=1,norm_fn='Batchnorm',activ_fn='tanh')]
        self.rec_layers=nn.ModuleList(layers2)

        ##Extracter
        ##
        layers3=[]
        n_in=3
        for i in range(0,extract_layers):
            n_out=extract_dim*(2**i)
            print(n_in,n_out)
            layers3.append(Conv2dBlockWithRes(n_in,n_out,Norm='Batch'))
            n_in=n_out
        #3->16->32->64->128->256
        self.extract=nn.ModuleList(layers3)


    def encoder(self,x):
        z=x
        zs=[]
        for layer in self.encode:
            z=layer(z)
            zs.append(z)
        return zs

    def decoder(self,zs,a):
        a_tile=a.view(a.size(0),-1,1,1).repeat(1,1,self.final_size,self.final_size)
        z=torch.cat([zs[-1],a_tile],dim=1)
        for i, layer in enumerate(self.dec_layers):
            z=layer(z)
            if self.shortcut_layers>i:
                z=torch.cat([z,zs[len(self.dec_layers)-2-i]],dim=1)
        return z

    def extracter(self,x):
        z=x
        for layer in self.extract:
            z=layer(z)
        return z

    def reconstructurer(self,zs,a):
        a_tile = a.view(a.size(0), -1, self.final_size, self.final_size)
        z=torch.cat([zs[-1],a_tile],dim=1)
        for i, layer in enumerate(self.rec_layers):
            z=layer(z)
            if self.shortcut_layers>i:
                z=torch.cat([z,zs[len(self.rec_layers)-2-i]],dim=1)
                #concat 1024 and 512
        return z

    def forward(self,x,trg=0,style=0,mode='NONE'):
        if(mode=='NONE'):
            print("Forward mistake!")
        if mode == 'enc-dec':
            return self.decoder(self.encoder(x), trg)
        if mode == 'ext':
            return self.extracter(x)#提取模式
        if mode == 'rec':
            return self.reconstructurer(self.encoder(x),style)#重建模式


def add_normalization_1d(layers,norm_fn,n_out):
    if norm_fn=='Batchnorm':
        layers.append(nn.BatchNorm1d(n_out))
    elif norm_fn=='Instancenorm':
        layers.append(nn.InstanceNorm1d(n_out))
    elif norm_fn=='none':
        pass
class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out, norm_fn='none', acti_fn='none'):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(n_in, n_out, bias=(norm_fn == 'none'))]
        add_normalization_1d(layers, norm_fn, n_out)
        add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Discriminators(nn.Module):
    def __init__(self, dim=64,
                 fc_dim=1024, n_attr=2,n_layers=5, img_size=128):
        super(Discriminators, self).__init__()
        self.f_size = img_size // 2 ** n_layers
        layers = []
        n_in = 3
        for i in range(0,n_layers):
            n_out=dim*(2**i)
            layers.append(Conv2dBlock(n_in,n_out,Norm='Instance'))
            n_in=n_out
        self.conv=nn.Sequential(*layers)
        self.fc_adv=nn.Sequential(LinearBlock(1024*self.f_size*self.f_size,fc_dim,'Instancenorm',acti_fn='lrelu'),
                             LinearBlock(fc_dim,1,'none','none')
            )
        self.fc_cls=nn.Sequential(LinearBlock(1024*self.f_size*self.f_size,fc_dim,'Instancenorm',acti_fn='lrelu'),
                             LinearBlock(fc_dim,n_attr,'none','none')
            )
    def forward(self,x):
        h=self.conv(x)
        h=h.view(h.size(0),-1)
        return self.fc_adv(h),self.fc_cls(h)



import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
import os
#import matplotlib.pyplot as plt
import random
from torchvision import transforms
from PIL import Image
Transformer=transforms.Compose([
    transforms.CenterCrop(128),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

class CelebA(Dataset):
    def __init__(self,image_dir,attr_path,select_attrs,transform,mode):
        self.image_dir=image_dir#mode决定数据集提供测试集还是训练集
        self.attr_path=attr_path
        self.select_attrs=select_attrs
        self.transform=transform
        self.mode=mode
        self.train_set=[]
        self.test_set=[]
        self.sample_set=[]
        self.attr2id={}
        self.id2attr={}
        self.preprocess()

        if mode=='train':
            self.num_images=len(self.train_set)
        else:
            self.num_images=len(self.test_set)

        print("{} Dataset Ready:{} images".format(self.mode,self.num_images))

    def preprocess(self):
        lines=[line.rstrip() for line in open(self.attr_path,'r')]
        all_attr_names=lines[1].split(sep=' ')
        for i, attr_name in enumerate(all_attr_names):
            self.attr2id[attr_name]=i
            self.id2attr[i]=attr_name

        for i,line in enumerate(lines):
            if i<2:
                continue
            split=line.split()
            filename=split[0]
            values=split[1:]
            for j in range(0,len(values)):
                if values[j]=='-1':
                    values[j]=0
                else:
                    values[j]=1
            label=[]
            for j,selected in enumerate(self.select_attrs):
                label.append(float(values[self.attr2id[selected]]))


            if(i+1)>6002:
                self.train_set.append([filename,label])
            else:
                self.test_set.append([filename,label])

            if filename in ['1.jpg','4.jpg','27.jpg','36.jpg','39.jpg','52.jpg','133.jpg','229.jpg']:
                self.sample_set.append([filename,label])


    def __getitem__(self,index):
        if self.mode=='Sample':
            dataset=self.sample_set
        else:
            dataset = self.train_set if self.mode == 'train' else self.test_set
        filename, label = dataset[index]
        image=Image.open(os.path.join(self.image_dir , filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_images



def get_loader(image_dir,attr_path,selected_attrs,batch_size=8,mode='train',num_workers=12):
    dataset=CelebA(image_dir,attr_path,selected_attrs,Transformer,mode)
    data_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    return data_loader



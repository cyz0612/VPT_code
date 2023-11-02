import pandas as pd
from pandas.core.frame import DataFrame
import cv2 as cv
import json
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from skimage.io import imread, imsave
import skimage
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from torchvision.datasets import ImageFolder
import re
from sklearn.metrics import roc_auc_score
from PIL import ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils import shuffle
import os

torch.cuda.empty_cache()


def getlabel(filename):
    fjson=open(filename)
    fjson=json.load(fjson)
    label_list=[]
    if fjson['flags']['取样-甲']==True:
        label_list.append(0)
    elif fjson['flags']['取样-乙']==True:
        label_list.append(1)
    elif fjson['flags']['取样-丙']==True:
        label_list.append(2)
        
    if fjson['flags']['制片-甲']==True:
        label_list.append(0)
    elif fjson['flags']['制片-乙']==True:
        label_list.append(1)
    elif fjson['flags']['制片-丙']==True:
        label_list.append(2)
        
    if fjson['flags']['染色-甲']==True:
        label_list.append(0)
    elif fjson['flags']['染色-乙']==True:
        label_list.append(1)
    elif fjson['flags']['染色-丙']==True:
        label_list.append(2)
        
    if fjson['flags']['封片-甲']==True:
        label_list.append(0)
    elif fjson['flags']['封片-乙']==True:
        label_list.append(1)
    elif fjson['flags']['封片-丙']==True:
        label_list.append(2)
    
    if len(label_list)!=4: 
        print("abnormal annotation !")
        print(filename)
    return label_list

def getlabel2(listlb):
    label_list=[]
    if listlb[0]=='甲':
        label_list.append(0)
    elif listlb[0]=='乙':
        label_list.append(1)
    elif listlb[0]=='丙':
        label_list.append(2)

    if listlb[1]=='甲':
        label_list.append(0)
    elif listlb[1]=='乙':
        label_list.append(1)
    elif listlb[1]=='丙':
        label_list.append(2)
  
    if listlb[2]=='甲':
        label_list.append(0)
    elif listlb[2]=='乙':
        label_list.append(1)
    elif listlb[2]=='丙':
        label_list.append(2)
    
    if listlb[3]=='甲':
        label_list.append(0)
    elif listlb[3]=='乙':
        label_list.append(1)
    elif listlb[3]=='丙':
        label_list.append(2)
    return label_list
    

def getsizename(size):
    if (size > 1024*1024*1024.0):
        numstr = str(size/(1024*1024*1024.0))
        sizename = numstr[:(numstr.index('.')+3)] + "GB"
    elif (size > 1024*1024.0):
        numstr = str(size/(1024*1024.0))
        sizename = numstr[:(numstr.index('.')+3)] + "MB"
    elif (size > 1024.0):
        numstr = str(size/1024.0)
        sizename = numstr[:(numstr.index('.')+3)] + "KB"
    else:
        sizename = str(size) + "Bytes"

    return sizename

class MultitaskLoss(nn.Module):
    def __init__(self):
        super(MultitaskLoss, self).__init__()
    def forward(self, output1, output2, output3,output4,target1, target2, target3,target4):
        diff_1 = output1-target1
        diff_2 = output2-target2
        diff_3 = output3-target3
        diff_4 = output4-target4

        error = torch.sqrt(diff_1 * diff_1 + diff_2 * diff_2 + diff_3 * diff_3 + diff_4 * diff_4)
        return torch.mean(error)

###########################_____________Load_Dataset___________________###################
class Cell(Dataset):
    def __init__(self,df,transform=None):
        self.num_cls = len(df)
        self.img_list = []
        for i in range(self.num_cls):
            self.img_list += [[df.iloc[i,0],df.iloc[i,1]]]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

#         img_path = self.img_list[idx][1]
        image = self.img_list[idx][0].convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label1': int(self.img_list[idx][1][0]),
                  'label2': int(self.img_list[idx][1][1]),
                  'label3': int(self.img_list[idx][1][2]),
                  'label4': int(self.img_list[idx][1][3])}
        return sample

alpha = None
alpha_name = f'{alpha}'
device = 'cuda'

class ResNet50(nn.Module):
    def __init__(self,**kwargs):
        super(ResNet50,self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.base=resnet50
        self.fc1 = nn.Linear(1000,3)
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(1000,3)
        self.fc3 = nn.Linear(1000,3)
        self.fc4 = nn.Linear(1000,3)


    def forward(self,x):
        x = self.base(x)
        out1=self.fc1(x)
        out2=self.fc2(x)
        out3=self.fc3(x)
        out4=self.fc4(x)
        #out = self.relu(out)
    
        return out1,out2,out3,out4
        
#################################_____________Train_Step_____________########################
def train(optimizer, epoch):
    
    model.train()
    
    train_loss = 0
    train_correct = 0
    
    for batch_index, batch_samples in enumerate(train_loader):
        
        data, target1, target2 ,target3, target4 = batch_samples['img'].to(device), batch_samples['label1'].to(device), batch_samples['label2'].to(device),batch_samples['label3'].to(device),batch_samples['label4'].to(device)

        optimizer.zero_grad()
        output1,output2,output3,output4= model(data)
        #criteria = nn.CrossEntropyLoss()
        criteria=nn.CrossEntropyLoss()
        #loss = criteria(output1, target1.long())
        #loss += criteria(output2, target2.long())
        #loss += criteria(output3, target3.long())
        #loss += criteria(output4, target4.long())
        #train_loss += loss

        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        #loss = criteria(output1,output2,output3,output4,target1.long(),target2.long(),target3.long(),target4.long())
        
        loss = criteria(output1, target1.long())
        loss += criteria(output2, target2.long())
        loss += criteria(output3, target3.long())
        loss += criteria(output4, target4.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss+=loss
        
        pred1 = output1.argmax(dim=1, keepdim=True)
        pred2 = output2.argmax(dim=1, keepdim=True)
        pred3 = output3.argmax(dim=1, keepdim=True)
        pred4 = output4.argmax(dim=1, keepdim=True)
        
        train_correct += pred1.eq(target1.long().view_as(pred1)).sum().item()
        train_correct += pred2.eq(target2.long().view_as(pred2)).sum().item()
        train_correct += pred3.eq(target3.long().view_as(pred3)).sum().item()
        train_correct += pred4.eq(target4.long().view_as(pred4)).sum().item()
    

        if batch_index % (0.5*bs) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))
    
    if epoch%10==0: torch.save(model.state_dict(),'/mnt/data/chenyuze/project2/code/checkpoint/new_mt/'+'20221220_r50_epoch%.1d'%epoch+'_224_bs=64.pth')
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss/len(train_loader.dataset), train_correct, 4*len(train_loader.dataset),
        100.0 * train_correct / (4*len(train_loader.dataset))))
    
    return train_loss/len(train_loader.dataset)


##################################___________Validation_Step_________________###############################
def val(epoch):
    
    model.eval()
    test_loss = 0
    correct = 0
    results = []
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    
    
    criteria = nn.CrossEntropyLoss()
    with torch.no_grad():
        tpr_list = []
        fpr_list = []
        
        predlist=[]
        scorelist=[]
        targetlist=[]
        
        qy_pred,zp_pred,rs_pred,fp_pred=[],[],[],[]
        qy_score,zp_score,rs_score,fp_score=[],[],[],[]
        qy_tg,zp_tg,rs_tg,fp_tg=[],[],[],[]
        for batch_index, batch_samples in enumerate(val_loader):
            data, target1, target2 ,target3, target4 = batch_samples['img'].to(device), batch_samples['label1'].to(device), batch_samples['label2'].to(device),batch_samples['label3'].to(device),batch_samples['label4'].to(device)
            
            output1,output2,output3,output4= model(data)
            
            test_loss += criteria(output1, target1.long())
            test_loss += criteria(output2, target2.long())
            test_loss += criteria(output3, target3.long())
            test_loss += criteria(output4, target4.long())
            #test_loss += criteria(output1,output2,output3,output4,target1.long(),target2.long(),target3.long(),target4.long())
            
            score1 = F.softmax(output1, dim=1)
            score2 = F.softmax(output2, dim=1)
            score3 = F.softmax(output3, dim=1)
            score4 = F.softmax(output4, dim=1)
            
            pred1 = output1.argmax(dim=1, keepdim=True)
            pred2 = output2.argmax(dim=1, keepdim=True)
            pred3 = output3.argmax(dim=1, keepdim=True)
            pred4 = output4.argmax(dim=1, keepdim=True)

            correct += pred1.eq(target1.long().view_as(pred1)).sum().item()
            correct += pred2.eq(target2.long().view_as(pred2)).sum().item()
            correct += pred3.eq(target3.long().view_as(pred3)).sum().item()
            correct += pred4.eq(target4.long().view_as(pred4)).sum().item()

            qyt=target1.long().cpu().numpy()
            zpt=target2.long().cpu().numpy()
            rst=target3.long().cpu().numpy()
            fpt=target4.long().cpu().numpy()
            
            qyp=pred1.cpu().numpy()
            zpp=pred2.cpu().numpy()
            rsp=pred3.cpu().numpy()
            fpp=pred4.cpu().numpy()
            
            qys=score1.cpu().numpy()[:,1]
            zps=score2.cpu().numpy()[:,1]
            rss=score3.cpu().numpy()[:,1]
            fps=score4.cpu().numpy()[:,1]
            
            qy_pred=np.append(qy_pred,qyp)
            zp_pred=np.append(zp_pred,zpp)
            rs_pred=np.append(rs_pred,rsp)
            fp_pred=np.append(fp_pred,fpp)
            
            qy_score=np.append(qy_score,qys)
            zp_score=np.append(zp_score,zps)
            rs_score=np.append(rs_score,rss)
            fp_score=np.append(fp_score,fps)
            
            qy_tg=np.append(qy_tg,qyt)
            zp_tg=np.append(zp_tg,zpt)
            rs_tg=np.append(rs_tg,rst)
            fp_tg=np.append(fp_tg,fpt)
            
    
    predlist.append([qy_pred,zp_pred,rs_pred,fp_pred])
    scorelist.append([qy_score,zp_score,rs_score,fp_score])
    targetlist.append([qy_tg,zp_tg,rs_tg,fp_tg])
           
    return targetlist, scorelist, predlist,test_loss/len(val_loader.dataset)
    
    
###########################_________Test_Step______________#############################
def test(epoch):
    
    model.eval()
    test_loss = 0
    correct = 0
    results = []
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    
    
    criteria = nn.CrossEntropyLoss()
    with torch.no_grad():
        tpr_list = []
        fpr_list = []
        
        predlist=[]
        scorelist=[]
        targetlist=[]
        qy_pred,zp_pred,rs_pred,fp_pred=[],[],[],[]
        qy_score,zp_score,rs_score,fp_score=[],[],[],[]
        qy_tg,zp_tg,rs_tg,fp_tg=[],[],[],[]
        #qyt,zpt,rst,fpt=[],[],[],[]
        #qyp,zpp,rsp,fpp=[],[],[],[]
        #qys,zps,rss,fps=[],[],[],[]
        for batch_index, batch_samples in enumerate(test_loader):
            data, target1, target2 ,target3, target4 = batch_samples['img'].to(device), batch_samples['label1'].to(device), batch_samples['label2'].to(device),batch_samples['label3'].to(device),batch_samples['label4'].to(device)
            
            output1,output2,output3,output4= model(data)
            
            test_loss += criteria(output1, target1.long())
            test_loss += criteria(output2, target2.long())
            test_loss += criteria(output3, target3.long())
            test_loss += criteria(output4, target4.long())
            #test_loss += criteria(output1,output2,output3,output4,target1.long(),target2.long(),target3.long(),target4.long())
            
            score1 = F.softmax(output1, dim=1)
            score2 = F.softmax(output2, dim=1)
            score3 = F.softmax(output3, dim=1)
            score4 = F.softmax(output4, dim=1)
            
            pred1 = output1.argmax(dim=1, keepdim=True)
            pred2 = output2.argmax(dim=1, keepdim=True)
            pred3 = output3.argmax(dim=1, keepdim=True)
            pred4 = output4.argmax(dim=1, keepdim=True)

            correct += pred1.eq(target1.long().view_as(pred1)).sum().item()
            correct += pred2.eq(target2.long().view_as(pred2)).sum().item()
            correct += pred3.eq(target3.long().view_as(pred3)).sum().item()
            correct += pred4.eq(target4.long().view_as(pred4)).sum().item()

            qyt=target1.long().cpu().numpy()
            zpt=target2.long().cpu().numpy()
            rst=target3.long().cpu().numpy()
            fpt=target4.long().cpu().numpy()
            
            qyp=pred1.cpu().numpy()
            zpp=pred2.cpu().numpy()
            rsp=pred3.cpu().numpy()
            fpp=pred4.cpu().numpy()
            
            qys=score1.cpu().numpy()[:,1]
            zps=score2.cpu().numpy()[:,1]
            rss=score3.cpu().numpy()[:,1]
            fps=score4.cpu().numpy()[:,1]
            
            qy_pred=np.append(qy_pred,qyp)
            zp_pred=np.append(zp_pred,zpp)
            rs_pred=np.append(rs_pred,rsp)
            fp_pred=np.append(fp_pred,fpp)
            
            qy_score=np.append(qy_score,qys)
            zp_score=np.append(zp_score,zps)
            rs_score=np.append(rs_score,rss)
            fp_score=np.append(fp_score,fps)
            
            qy_tg=np.append(qy_tg,qyt)
            zp_tg=np.append(zp_tg,zpt)
            rs_tg=np.append(rs_tg,rst)
            fp_tg=np.append(fp_tg,fpt)
            
    
    predlist.append([qy_pred,zp_pred,rs_pred,fp_pred])
    scorelist.append([qy_score,zp_score,rs_score,fp_score])
    targetlist.append([qy_tg,zp_tg,rs_tg,fp_tg])
            
    return targetlist, scorelist, predlist

if __name__=='__main__':
    img_list1=[]
    for filename in glob.glob('/mnt/data/chenyuze/project2/data/train/*.jpg'): 
        im=Image.open(filename)
        #im=im.resize((1393,1044)).crop((0,0,1024,1024))
        lb=getlabel(filename[0:-4]+'.json')
        img_list1.append([im,lb])
    #train_df=DataFrame(img_list1)
    '''
    print("取样：",sum(train_df.iloc[:,1]==0))
    print(sum(train_df.iloc[:,1]==1))
    print(sum(train_df.iloc[:,1]==2))
    print("制片：",sum(train_df.iloc[:,2]==0))
    print(sum(train_df.iloc[:,2]==1))
    print(sum(train_df.iloc[:,2]==2))
    print("染色：",sum(train_df.iloc[:,3]==0))
    print(sum(train_df.iloc[:,3]==1))
    print(sum(train_df.iloc[:,3]==2))
    print("封片：",sum(train_df.iloc[:,4]==0))
    print(sum(train_df.iloc[:,4]==1))
    print(sum(train_df.iloc[:,4]==2))
    '''
    img_list2=[]
    for filename in glob.glob('/mnt/data/chenyuze/project2/data/val/*.jpg'): 
        im=Image.open(filename)
        #im=im.resize((1393,1044)).crop((0,0,1024,1024))
        lb=getlabel(filename[0:-4]+'.json')
        img_list2.append([im,lb])
    
    '''
    print("取样：",sum(valset_df.iloc[:,1]==0))
    print(sum(valset_df.iloc[:,1]==1))
    print(sum(valset_df.iloc[:,1]==2))
    print("制片：",sum(valset_df.iloc[:,2]==0))
    print(sum(valset_df.iloc[:,2]==1))
    print(sum(valset_df.iloc[:,2]==2))
    print("染色：",sum(valset_df.iloc[:,3]==0))
    print(sum(valset_df.iloc[:,3]==1))
    print(sum(valset_df.iloc[:,3]==2))
    print("封片：",sum(valset_df.iloc[:,4]==0))
    print(sum(valset_df.iloc[:,4]==1))
    print(sum(valset_df.iloc[:,4]==2))
    '''
    img_list3=[]
    for filename in glob.glob('/mnt/data/chenyuze/project2/data/test/*.jpg'): 
        im=Image.open(filename)
        #im=im.resize((1393,1044)).crop((0,0,1024,1024))
        lb=getlabel(filename[0:-4]+'.json')
        img_list3.append([im,lb])
    
    '''
    print("取样：",sum(testset_df.iloc[:,1]==0))
    print(sum(testset_df.iloc[:,1]==1))
    print(sum(testset_df.iloc[:,1]==2))
    print("制片：",sum(testset_df.iloc[:,2]==0))
    print(sum(testset_df.iloc[:,2]==1))
    print(sum(testset_df.iloc[:,2]==2))
    print("染色：",sum(testset_df.iloc[:,3]==0))
    print(sum(testset_df.iloc[:,3]==1))
    print(sum(testset_df.iloc[:,3]==2))
    print("封片：",sum(testset_df.iloc[:,4]==0))
    print(sum(testset_df.iloc[:,4]==1))
    print(sum(testset_df.iloc[:,4]==2))
    '''
    #label_df=pd.read_excel('/mnt/data/chenyuze/project2/20211229_51data_bumanyi/70个不满意样本.xlsx')
    for f in glob.glob('/mnt/nas/data/zhikong/new_train_data/*'):
        folder=f
        lb=[0,0,0,0]
        L={}
        k={}
        d = folder+'/torch/'
        files = os.listdir(d)    
        filenames = os.listdir(d)
        for filename in filenames:
            fullpath = os.path.join(d, filename)
          
            L[filename] = os.path.getsize(fullpath)
        
        knum=50
        k = sorted(L.items(), key=lambda L:L[1], reverse = True)
        for j in range(knum):
            im=Image.open(d+k[j][0])
            if j<30:             
                img_list1.append([im,lb])
            elif j<40:
                img_list2.append([im,lb])
            else:
                img_list3.append([im,lb])
                
    new_folder=['C22101900067C101_20','C22102300214C101_20','C22102200199C101_20','C22102100061C101_20']
    for folder1 in new_folder:
        lb=[1,0,0,0]
        L={}
        k={}
        d = '/mnt/nas/data/zhikong/20221027_zhikong_verify_133/'+folder1+'/torch/'
        files = os.listdir(d)    
        filenames = os.listdir(d)
        for filename in filenames:
            fullpath = os.path.join(d, filename)
          
            L[filename] = os.path.getsize(fullpath)
        
        knum=20
        k = sorted(L.items(), key=lambda L:L[1], reverse = True)
        for j in range(knum):
            im=Image.open(d+k[j][0])
            if j<14:             
                img_list1.append([im,lb])
            elif j<17:
                img_list2.append([im,lb])
            else:
                img_list3.append([im,lb])
    
    lb=[0,0,0,0]
    L={}
    k={}
    d = '/mnt/nas/data/zhikong/20221027_zhikong_verify_133/C22102300110C101_20/torch/'
    files = os.listdir(d)    
    filenames = os.listdir(d)
    for filename in filenames:
        fullpath = os.path.join(d, filename)
      
        L[filename] = os.path.getsize(fullpath)
    
    knum=5
    k = sorted(L.items(), key=lambda L:L[1], reverse = True)
    for j in range(knum):
        im=Image.open(d+k[j][0])           
        img_list1.append([im,lb])
    
    lb=[1,0,0,0]
    L={}
    k={}
    d = '/mnt/nas/data/zhikong/20221027_zhikong_verify_133/C22102300615C101_20/torch/'
    files = os.listdir(d)    
    filenames = os.listdir(d)
    for filename in filenames:
        fullpath = os.path.join(d, filename)
      
        L[filename] = os.path.getsize(fullpath)
    
    k = sorted(L.items(), key=lambda L:L[1], reverse = True)
    for j in range(10,30):
        im=Image.open(d+k[j][0])           
        if j<24:             
            img_list1.append([im,lb])
        elif j<27:
            img_list2.append([im,lb])
        else:
            img_list3.append([im,lb])

    lb=[0,0,0,0]
    L={}
    k={}
    d = '/mnt/nas/data/zhikong/20221027_zhikong_verify_133/C22102500327C101_20/torch/'
    files = os.listdir(d)    
    filenames = os.listdir(d)
    for filename in filenames:
        fullpath = os.path.join(d, filename)
      
        L[filename] = os.path.getsize(fullpath)
    
    k = sorted(L.items(), key=lambda L:L[1], reverse = True)
    for j in range(20):
        im=Image.open(d+k[j][0])           
        if j<14:             
            img_list1.append([im,lb])
        elif j<17:
            img_list2.append([im,lb])
        else:
            img_list3.append([im,lb])

    train_df=DataFrame(img_list1)
    # train_df=shuffle(train_df)
    valset_df=DataFrame(img_list2)
    testset_df=DataFrame(img_list3)

    lb=[0,0,0,0]
    L={}
    k={}
    d = '/mnt/nas/data/zhikong/20221219_wupan_zhikong_37/C22103100311C101_20/torch/'
    files = os.listdir(d)    
    filenames = os.listdir(d)
    for filename in filenames:
        fullpath = os.path.join(d, filename)
      
        L[filename] = os.path.getsize(fullpath)
    
    k = sorted(L.items(), key=lambda L:L[1], reverse = True)
    for j in range(30):
        im=Image.open(d+k[j][0])           
        if j<20:             
            img_list1.append([im,lb])
        elif j<25:
            img_list2.append([im,lb])
        else:
            img_list3.append([im,lb])
    
    lb=[0,0,0,0]
    L={}
    k={}
    d = '/mnt/nas/data/zhikong/20221219_wupan_zhikong_37/C22103100404C101_20/torch/'
    files = os.listdir(d)    
    filenames = os.listdir(d)
    for filename in filenames:
        fullpath = os.path.join(d, filename)
      
        L[filename] = os.path.getsize(fullpath)
    
    k = sorted(L.items(), key=lambda L:L[1], reverse = True)
    for j in range(30):
        im=Image.open(d+k[j][0])           
        if j<20:             
            img_list1.append([im,lb])
        elif j<25:
            img_list2.append([im,lb])
        else:
            img_list3.append([im,lb])

    train_df=DataFrame(img_list1)
    # train_df=shuffle(train_df)
    valset_df=DataFrame(img_list2)
    testset_df=DataFrame(img_list3)

    lb=[0,0,0,0]
    L={}
    k={}
    d = '/mnt/nas/data/zhikong/20221219_wupan_zhikong_37/C22110100028C101_20/torch/'
    files = os.listdir(d)    
    filenames = os.listdir(d)
    for filename in filenames:
        fullpath = os.path.join(d, filename)
      
        L[filename] = os.path.getsize(fullpath)
    
    k = sorted(L.items(), key=lambda L:L[1], reverse = True)
    for j in range(30):
        im=Image.open(d+k[j][0])           
        if j<20:             
            img_list1.append([im,lb])
        elif j<25:
            img_list2.append([im,lb])
        else:
            img_list3.append([im,lb])

    train_df=DataFrame(img_list1)
    # train_df=shuffle(train_df)
    valset_df=DataFrame(img_list2)
    testset_df=DataFrame(img_list3)

    lb=[0,0,0,0]
    L={}
    k={}
    d = '/mnt/nas/data/zhikong/20221219_wupan_zhikong_37/C22110100065C101_20/torch/'
    files = os.listdir(d)    
    filenames = os.listdir(d)
    for filename in filenames:
        fullpath = os.path.join(d, filename)
      
        L[filename] = os.path.getsize(fullpath)
    
    k = sorted(L.items(), key=lambda L:L[1], reverse = True)
    for j in range(30):
        im=Image.open(d+k[j][0])           
        if j<20:             
            img_list1.append([im,lb])
        elif j<25:
            img_list2.append([im,lb])
        else:
            img_list3.append([im,lb])

    train_df=DataFrame(img_list1)
    # train_df=shuffle(train_df)
    valset_df=DataFrame(img_list2)
    testset_df=DataFrame(img_list3)

    lb=[0,0,0,0]
    L={}
    k={}
    d = '/mnt/nas/data/zhikong/20221219_wupan_zhikong_37/C22110100072C101_20/torch/'
    files = os.listdir(d)    
    filenames = os.listdir(d)
    for filename in filenames:
        fullpath = os.path.join(d, filename)
      
        L[filename] = os.path.getsize(fullpath)
    
    k = sorted(L.items(), key=lambda L:L[1], reverse = True)
    for j in range(30):
        im=Image.open(d+k[j][0])           
        if j<20:             
            img_list1.append([im,lb])
        elif j<25:
            img_list2.append([im,lb])
        else:
            img_list3.append([im,lb])

    train_df=DataFrame(img_list1)
    # train_df=shuffle(train_df)
    valset_df=DataFrame(img_list2)
    testset_df=DataFrame(img_list3)

    lb=[1,0,0,0]
    L={}
    k={}
    d = '/mnt/nas/data/zhikong/20221219_wupan_zhikong_37/C22110100203C101_20/torch/'
    files = os.listdir(d)    
    filenames = os.listdir(d)
    for filename in filenames:
        fullpath = os.path.join(d, filename)
      
        L[filename] = os.path.getsize(fullpath)
    
    k = sorted(L.items(), key=lambda L:L[1], reverse = True)
    for j in range(30):
        im=Image.open(d+k[j][0])           
        if j<20:             
            img_list1.append([im,lb])
        elif j<25:
            img_list2.append([im,lb])
        else:
            img_list3.append([im,lb])

    train_df=DataFrame(img_list1)
    # train_df=shuffle(train_df)
    valset_df=DataFrame(img_list2)
    testset_df=DataFrame(img_list3)

    lb=[0,0,0,0]
    L={}
    k={}
    d = '/mnt/nas/data/zhikong/20221219_wupan_zhikong_37/C22110100338C101_20/torch/'
    files = os.listdir(d)    
    filenames = os.listdir(d)
    for filename in filenames:
        fullpath = os.path.join(d, filename)
      
        L[filename] = os.path.getsize(fullpath)
    
    k = sorted(L.items(), key=lambda L:L[1], reverse = True)
    for j in range(20):
        im=Image.open(d+k[j][0])           
        if j<14:             
            img_list1.append([im,lb])
        elif j<17:
            img_list2.append([im,lb])
        else:
            img_list3.append([im,lb])

    train_df=DataFrame(img_list1)
    # train_df=shuffle(train_df)
    valset_df=DataFrame(img_list2)
    testset_df=DataFrame(img_list3)

    lb=[1,0,0,0]
    L={}
    k={}
    d = '/mnt/nas/data/zhikong/20221219_wupan_zhikong_37/C22110400431C101_20/torch/'
    files = os.listdir(d)    
    filenames = os.listdir(d)
    for filename in filenames:
        fullpath = os.path.join(d, filename)
      
        L[filename] = os.path.getsize(fullpath)
    
    k = sorted(L.items(), key=lambda L:L[1], reverse = True)
    for j in range(20):
        im=Image.open(d+k[j][0])           
        if j<14:             
            img_list1.append([im,lb])
        elif j<17:
            img_list2.append([im,lb])
        else:
            img_list3.append([im,lb])

    train_df=DataFrame(img_list1)
    # train_df=shuffle(train_df)
    valset_df=DataFrame(img_list2)
    testset_df=DataFrame(img_list3)

    '''
    dlen=len(dataset_df)
    train_df=dataset_df.iloc[0:int(np.floor(0.9*dlen)),:]
    valset_df=dataset_df.iloc[int(np.ceil(0.9*dlen)):int(np.floor(0.95*dlen)),:]
    testset_df=dataset_df.iloc[int(np.ceil(0.95*dlen)):dlen,:]
    #testset_df=DataFrame(img_list3)
    '''
    ### read all images:
    '''
    img_list=[]
    for filename in glob.glob('/mnt/data/chenyuze/project2/data/zhikong/*.jpg'): 
        im=Image.open(filename)
        img_list.append([im,getlabel(filename[0:-4]+'.json')])
    all_images_df=DataFrame(img_list)
    print(all_images_df)
    
    # k-fold cross-validation:
    k=5
    score=[0]*k
    for i in range(k):
    '''
    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                         std=[0.33165374, 0.33165374, 0.33165374])
    
    transformer = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation((0,90), resample=False, expand=False, center=None, fill=None),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize
    ])
    
    batchsize=64
    trainset =Cell(df=train_df,transform= transformer)
    valset=Cell(df=valset_df,transform=transformer)
    testset=Cell(df=testset_df,transform=transformer)
    
    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=True)
    
            
    model = ResNet50().cuda()
    modelname = 'ResNet50'
    #model.load_state_dict(torch.load('/mnt/data/chenyuze/project2/code/checkpoint/new_mt/ResNet50_epoch30_224_bs=64.pth'))

    # train
    print('\n\n\n\n\n\n\n\n\n\n\n--------------------------------------------Multi-task Learning--------------------------------------------\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
    print('-------------------------------train------------------------------------')
    
    bs=batchsize
    votenum = 1
    import warnings
    warnings.filterwarnings('ignore')
    
    r_list = []
    p_list = []
    acc_list = []
    AUC_list = []
    vote_pred = np.zeros(valset.__len__())
    vote_score = np.zeros(valset.__len__())
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
                                                 
    scheduler = StepLR(optimizer, step_size=1)
    
    total_epoch = 50
    train_num=range(1,total_epoch+1)
    train_loss=[]
    val_loss=[]
    for epoch in range(1, total_epoch+1):
        loss=train(optimizer, epoch)
        train_loss.append(loss)
        
        targetlist, scorelist, predlist,loss2 = val(epoch)
        val_loss.append(loss2)
        #print('target',targetlist)
        #print('score',scorelist)
        #print('predict',predlist)
        #vote_pred = vote_pred + predlist 
        #vote_score = vote_score + scorelist 
    
        if epoch % votenum == 0:
            
            # major vote
            #vote_pred[vote_pred <= (votenum/2)] = 0
            #vote_pred[vote_pred > (votenum/2)] = 1
            #vote_score = vote_score/votenum
            
            #print('vote_pred', vote_pred)
            print("----------------------------取样：")
            pre = (predlist[0][0] == targetlist[0][0]).sum()
            print('accuracy:',pre,'       ',len(predlist[0][0]),'       ',pre/len(predlist[0][0]))
            #print("target:\n")
            #print("甲：",(targetlist[0][0]==0).sum(),"\n乙：",(targetlist[0][0]==1).sum(),"\n丙：",(targetlist[0][0]==2).sum())
            #print("predict:\n")
            #print("甲：",(predlist[0][0]==0).sum(),"\n乙：",(predlist[0][0]==1).sum(),"\n丙：",(predlist[0][0]==2).sum())
            print("----------------------------制片：")
            pre = (predlist[0][1] == targetlist[0][1]).sum()
            print('accuracy:',pre,'       ',len(predlist[0][1]),'       ',pre/len(predlist[0][1]))
            #print("target:\n")
            #print("甲：",(targetlist[0][1]==0).sum(),"\n乙：",(targetlist[0][1]==1).sum(),"\n丙：",(targetlist[0][1]==2).sum())
            #print("predict:\n")
            #print("甲：",(predlist[0][1]==0).sum(),"\n乙：",(predlist[0][1]==1).sum(),"\n丙：",(predlist[0][1]==2).sum())
            print("----------------------------染色：")
            pre = (predlist[0][2] == targetlist[0][2]).sum()
            print('accuracy:',pre,'       ',len(predlist[0][2]),'       ',pre/len(predlist[0][2]))
            #print("target:\n")
            #print("甲：",(targetlist[0][2]==0).sum(),"\n乙：",(targetlist[0][2]==1).sum(),"\n丙：",(targetlist[0][2]==2).sum())
            #print("predict:\n")
            #print("甲：",(predlist[0][2]==0).sum(),"\n乙：",(predlist[0][2]==1).sum(),"\n丙：",(predlist[0][2]==2).sum())
            print("----------------------------封片：")
            pre = (predlist[0][3] == targetlist[0][3]).sum()
            print('accuracy:',pre,'       ',len(predlist[0][3]),'       ',pre/len(predlist[0][3]))
            #print("target:\n")
            #print("甲：",(targetlist[0][3]==0).sum(),"\n乙：",(targetlist[0][3]==1).sum(),"\n丙：",(targetlist[0][3]==2).sum())
            #print("predict:\n")
            #print("甲：",(predlist[0][3]==0).sum(),"\n乙：",(predlist[0][3]==1).sum(),"\n丙：",(predlist[0][3]==2).sum())

    
    

    
    # plt.plot(train_num,train_loss, '-',color='red',label=u'train_loss')
    # plt.plot(train_num,val_loss, '-',color='blue',label=u'validation_loss')
    # plt.xlabel('epoches')
    # plt.ylabel('loss')
    # plt.title('Loss Curve')
    # plt.legend()
    # plt.show()
    # plt.savefig("Average_loss_mt_14000.jpg")
    
    
    # test
    print('-------------------------------test------------------------------------')
    bs = batchsize
    import warnings
    warnings.filterwarnings('ignore')
    
    #vote_pred = np.zeros(testset.__len__())
    #vote_score = np.zeros(testset.__len__())
    
    epoch=30
    targetlist, scorelist, predlist = test(epoch)
    print("----------------------------取样：")
    pre = (predlist[0][0] == targetlist[0][0]).sum()
    print('accuracy:',pre,'       ',len(predlist[0][0]),'       ',pre/len(predlist[0][0]))
    print("target:\n")
    print("甲：",(targetlist[0][0]==0).sum(),"\n乙：",(targetlist[0][0]==1).sum(),"\n丙：",(targetlist[0][0]==2).sum())
    print("predict:\n")
    print("甲：",(predlist[0][0]==0).sum(),"\n乙：",(predlist[0][0]==1).sum(),"\n丙：",(predlist[0][0]==2).sum())
    print("----------------------------制片：")
    pre = (predlist[0][1] == targetlist[0][1]).sum()
    print('accuracy:',pre,'       ',len(predlist[0][1]),'       ',pre/len(predlist[0][1]))
    print("target:\n")
    print("甲：",(targetlist[0][1]==0).sum(),"\n乙：",(targetlist[0][1]==1).sum(),"\n丙：",(targetlist[0][1]==2).sum())
    print("predict:\n")
    print("甲：",(predlist[0][1]==0).sum(),"\n乙：",(predlist[0][1]==1).sum(),"\n丙：",(predlist[0][1]==2).sum())
    print("----------------------------染色：")
    pre = (predlist[0][2] == targetlist[0][2]).sum()
    print('accuracy:',pre,'       ',len(predlist[0][2]),'       ',pre/len(predlist[0][2]))
    print("target:\n")
    print("甲：",(targetlist[0][2]==0).sum(),"\n乙：",(targetlist[0][2]==1).sum(),"\n丙：",(targetlist[0][2]==2).sum())
    print("predict:\n")
    print("甲：",(predlist[0][2]==0).sum(),"\n乙：",(predlist[0][2]==1).sum(),"\n丙：",(predlist[0][2]==2).sum())
    print("----------------------------封片：")
    pre = (predlist[0][3] == targetlist[0][3]).sum()
    print('accuracy:',pre,'       ',len(predlist[0][3]),'       ',pre/len(predlist[0][3]))
    print("target:\n")
    print("甲：",(targetlist[0][3]==0).sum(),"\n乙：",(targetlist[0][3]==1).sum(),"\n丙：",(targetlist[0][3]==2).sum())
    print("predict:\n")
    print("甲：",(predlist[0][3]==0).sum(),"\n乙：",(predlist[0][3]==1).sum(),"\n丙：",(predlist[0][3]==2).sum())
    '''
    vote_pred = vote_pred + predlist 
    vote_score = vote_score + scorelist 
    
    TP = ((predlist == 1) & (targetlist == 1)).sum()
    
    TN = ((predlist == 0) & (targetlist == 0)).sum()
    FN = ((predlist == 0) & (targetlist == 1)).sum()
    FP = ((predlist == 1) & (targetlist == 0)).sum()
    
    print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
    print('TP+FP',TP+FP)
    p = TP / (TP + FP)
    print('precision',p)
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    print('recall',r)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('F1',F1)
    print('acc',acc)
    AUC = roc_auc_score(targetlist, vote_score,multi_class='ovr')
    print('AUC', AUC)
    '''
    
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    conf_mat = confusion_matrix(targetlist,predlist)
    sns.heatmap(conf_mat,annot=True,xticklabels=['0','1','2'],yticklabels=['0','1','2'])
    ax.set_title('confusion matrix -fengpian')
    ax.set_xlabel('predict')
    ax.set_ylabel('true label')
    plt.show()
    plt.savefig("confusion_matrix_fengpian.png")
    '''
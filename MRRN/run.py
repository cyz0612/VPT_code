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
from sklearn.utils import shuffle
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
import os

torch.cuda.empty_cache()
device='cuda'


###########################_____________Load_Dataset___________________###################
class Bone(Dataset):
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
                  'label': int(self.img_list[idx][1])}
        return sample


#################################_____________Train_Step_____________########################
def train(optimizer, epoch):
    
    model.train()
    
    train_loss = 0
    train_correct = 0
    
    for batch_index, batch_samples in enumerate(train_loader):
        
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

        optimizer.zero_grad()
        output,_ = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        criteria = nn.CrossEntropyLoss()
        # print("output:",output,output.shape)
        # print(pred)
        loss = criteria(output, target.long())

        train_loss += criteria(output, target.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        # print("pred:",pred)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()
    

        if batch_index % (10*bs) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))
    
    # if epoch%10==0: torch.save(model.state_dict(),'bone_resnet18.pth')
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    if train_correct / len(train_loader.dataset)>=0.99: torch.save(model.state_dict(),'bone_resnet18.pth')
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
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            
            output,_ = model(data)
            
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.long().view_as(pred)).sum().item()

            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
           
    return targetlist, scorelist, predlist
    
    
###########################_________Test_Step______________#############################
def test(epoch):
    
    model.eval()
    test_loss = 0
    correct = 0
    results = []
    
    criteria = nn.CrossEntropyLoss()
    with torch.no_grad():
        
        predlist=[]
        scorelist=[]
        targetlist=[]
        feature_list=[]
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            output,feature = model(data)
            print(feature.cpu().numpy().shape)
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
            feature_list.extend(feature.cpu().tolist())

    print(type(feature_list))
    return targetlist, scorelist, predlist,np.array(feature_list)

img_list1=pd.read_excel("img_label2.xlsx")
img_list2=pd.read_excel("xray_with_plan_photos_path.xlsx")
img_df=[]
for iii in range(len(img_list1)): 

    im=Image.open(img_list1.iloc[iii,0])
    # print(im.size)
    img_df.append([im,img_list1.iloc[iii,1]])

for jjj in range(len(img_list2)):
    im=Image.open(img_list1.iloc[iii,0])
    # print(im.size)
    img_df.append([im,img_list1.iloc[iii,1]])

img_df=DataFrame(img_df)
img_df=shuffle(img_df)
train_df=img_df[:450]
test_df=img_df[450:]


normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])

transformer = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    normalize
])


trainset =Bone(df=train_df,transform= transformer)
valset=Bone(df=test_df,transform=transformer)
testset=Bone(df=img_df,transform=transformer)

train_loader = DataLoader(trainset, batch_size=64, drop_last=False, shuffle=True)
val_loader = DataLoader(valset, batch_size=64, drop_last=False, shuffle=False)
# test_loader = DataLoader(testset, batch_size=16, drop_last=False, shuffle=False)
test_loader = DataLoader(testset, batch_size=64, drop_last=False, shuffle=False)

# alpha = None
# alpha_name = f'{alpha}'
# device = 'cuda'
# model = models.resnet50(pretrained=True).cuda()
# modelname = 'ResNet50'

class ResNet18(nn.Module):
    def __init__(self,**kwargs):
        super(ResNet18,self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.base=resnet18
        self.fc1 = nn.Linear(1000,128)
        self.fc2 = nn.Linear(128,4)
        self.relu=nn.ReLU()


    def forward(self,x):
        x1 = self.base(x)
        out=self.fc1(x1)
        out=self.relu(out)
        out1=self.fc2(out)
        out1=self.relu(out1)
        #out = self.relu(out)
    
        return out1,out

model = ResNet18().cuda()
modelname = 'ResNet18'
# model.load_state_dict(torch.load("bone_resnet18.pth"))

# train
# print('-----------------------------------封片------------------------------------\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('-------------------------------train------------------------------------')
batchsize=4
bs=batchsize
votenum = 4
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

total_epoch = 10
train_num=range(1,total_epoch+1)
avg_loss=[]
for epoch in range(1, total_epoch+1):
    loss=train(optimizer, epoch)
    avg_loss.append(loss)
    
    targetlist, scorelist, predlist = val(epoch)
    # print('target',targetlist)
    #print('score',scorelist)
    # print('predict',predlist)
    vote_pred = vote_pred + predlist 
    vote_score = vote_score + scorelist 

    if epoch % votenum == 0:
        
        # major vote
        vote_pred[vote_pred <= (votenum/2)] = 0
        vote_pred[vote_pred > (votenum/2)] = 1
        vote_score = vote_score/votenum
        
        #print('vote_pred', vote_pred)
        print('targetlist', targetlist)
        pre = (predlist == targetlist).sum()
        print('accuracy:',pre,'       ',len(predlist),'       ',pre/len(predlist))





# model.load_state_dict(torch.load("bone_resnet18.pth"))
# # test
print('-------------------------------test------------------------------------')
bs = 4
import warnings
warnings.filterwarnings('ignore')

epoch = 100
r_list = []
p_list = []
acc_list = []
AUC_list = []
# TP = 0
# TN = 0
# FN = 0
# FP = 0
vote_pred = np.zeros(testset.__len__())
vote_score = np.zeros(testset.__len__())


targetlist, scorelist, predlist,featlist = test(epoch)
print('target',targetlist)
# print('score',scorelist)
print('predict',predlist)
print("features:",featlist,featlist.shape)
np.save("xray_feature.npy",featlist)
pre = (predlist == targetlist).sum()
print('accuracy:',pre,'       ',len(predlist),'       ',pre/len(predlist))


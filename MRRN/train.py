# from model import Model
# import numpy as np
# import os
# import torch
# from torchvision.datasets import mnist
# from torch.nn import CrossEntropyLoss
# from torch.optim import SGD
# from torch.utils.data import DataLoader
# from torchvision.transforms import ToTensor

# if __name__ == '__main__':
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     batch_size = 256
#     train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
#     test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
#     # print(test_dataset[0][0].shape)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)
#     model = Model().to(device)
#     sgd = SGD(model.parameters(), lr=1e-1)
#     loss_fn = CrossEntropyLoss()
#     all_epoch = 100
#     prev_acc = 0
#     for current_epoch in range(all_epoch):
#         model.train()
#         for idx, (train_x, train_label) in enumerate(train_loader):
#             train_x = train_x.to(device)
#             # print(train_x.size())
#             train_label = train_label.to(device)
#             sgd.zero_grad()
#             predict_y = model(train_x.float())
#             loss = loss_fn(predict_y, train_label.long())
#             loss.backward()
#             sgd.step()

#         all_correct_num = 0
#         all_sample_num = 0
#         model.eval()
        
#         for idx, (test_x, test_label) in enumerate(test_loader):
#             test_x = test_x.to(device)
#             test_label = test_label.to(device)
#             predict_y = model(test_x.float()).detach()
#             predict_y =torch.argmax(predict_y, dim=-1)
#             current_correct_num = predict_y == test_label
#             all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
#             all_sample_num += current_correct_num.shape[0]
#         acc = all_correct_num / all_sample_num
#         print('accuracy: {:.3f}'.format(acc))
#         if not os.path.isdir("models"):
#             os.mkdir("models")
#         torch.save(model, 'models/mnist_{:.3f}.pkl'.format(acc))
#         # if np.abs(acc - prev_acc) < 1e-4:
#         #     break
#         prev_acc = acc
#     print("Model finished training")


from model import Model
import numpy as np
import os
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor
import pandas as pd
from pandas import DataFrame
import torch.nn as nn
import cv2
import torch.optim as optim

class ResNet18(nn.Module):
    def __init__(self,**kwargs):
        super(ResNet18,self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.base=resnet50
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

class Clinical(Dataset):
    def __init__(self,df,transform=None):
        self.num_cls = len(df)
        self.img_list = []
        for i in range(self.num_cls):
            self.img_list += [[df.iloc[i,0],df.iloc[i,1],df.iloc[i,2],df.iloc[i,3]]]
        # self.img_list=df
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

#         img_path = self.img_list[idx][1]
        # image = self.img_list[idx][0].convert('RGB')

        # if self.transform:
        #     image = self.transform(image)
        sample = {'sfz': self.img_list[idx][0],
                  'clinical': np.array(self.img_list[idx][1]),
                  'img_ft': np.array(self.img_list[idx][2]),
                  'plan': np.array(self.img_list[idx][3])}
        return sample

def train(optimizer, epoch,bs):
    model.train()
    train_loss = 0
    for batch_index, batch_samples in enumerate(train_loader):
        clinical_f,img_f,wb_plan= batch_samples['clinical'].cuda(),batch_samples['img_ft'].cuda(), batch_samples['plan'].cuda().to(torch.float)
        optimizer.zero_grad()
        output = model(clinical_f,img_f)
        # print("output:",output,"plan:",wb_plan)
        criteria = nn.MSELoss()
        loss = criteria(output,wb_plan)
        # print(type(loss))
        train_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if batch_index % (3) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]    Train Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))
    
    # if epoch%10==0: torch.save(model.state_dict(),'bone_resnet18.pth')
    print('Train set: Average loss: {:.4f}'.format(
        train_loss/len(train_loader.dataset)))
    # if train_correct / len(train_loader.dataset)>=0.99: torch.save(model.state_dict(),'bone_resnet18.pth')
    return train_loss/len(train_loader.dataset)

def test(epoch): 
    model.eval()
    test_loss = 0   
    criteria = nn.MSELoss()
    with torch.no_grad():
        predlist=[]
        scorelist=[]
        targetlist=[]
        feature_list=[]
        for _, batch_samples in enumerate(test_loader):
            sfz,clinical_f,img_f,wb_plan= batch_samples['sfz'],batch_samples['clinical'].cuda(),batch_samples['img_ft'].cuda(), batch_samples['plan'].cuda().to(torch.float)
            output = model(clinical_f,img_f)
            test_loss += criteria(output,wb_plan)
            targetcpu=wb_plan.cpu().numpy()
            predlist=np.append(predlist,output.cpu().numpy())

            targetlist=np.append(targetlist,targetcpu)
    return targetlist,predlist


if __name__ == '__main__':
    model1 = ResNet18().cuda()
    model1.load_state_dict(torch.load("bone_resnet18.pth"))

    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                        std=[0.33165374, 0.33165374, 0.33165374])

    transformer = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        normalize
    ])

    linchuang_data=pd.read_excel("linchuang.xlsx")
    for i in range(len(linchuang_data)):
        if linchuang_data.iloc[i,5]=='男':
            linchuang_data.iloc[i,5]=1
        else:
            linchuang_data.iloc[i,5]=0
    info_dict=[]
    for i in range(len(linchuang_data)):
        info_dict.append([linchuang_data.iloc[i,1],linchuang_data.iloc[i,2:].tolist()])
    info_dict=DataFrame(info_dict)
    info_dict.columns=["sfz","clinical"]
    im_feat_list=[]
    for s in range(len(info_dict)):
        sfz=info_dict.loc[s,"sfz"]
        xray_dir="./0308xray/"+sfz+"/术后/"
        if not os.path.exists(xray_dir):
            xray_dir=xray_dir.replace("后","前")
        # print(xray_dir)
        xray=os.listdir(xray_dir)[0]
        xray=os.path.join(xray_dir,xray)
        xray_img=Image.open(xray)
        xray_img=transformer(xray_img)
        xray_img=xray_img.unsqueeze(0)
        model1.eval()
        _,img_feature=model1(xray_img.cuda())
        im_feat_list.append(img_feature.cpu().detach().numpy()[0])
    info_dict["img_feature"]=im_feat_list

    drop_list=[]
    for i in range(len(info_dict)):
        sfz=info_dict.iloc[i,0]
        plan_dir="../Multidimensional-time-series-with-transformer-main/0307data/"+sfz+"/tb_sub_plan.xlsx"
        if not os.path.exists(plan_dir):
            drop_list.append(i)
    info_dict.drop(drop_list,axis=0,inplace=True)
    info_dict.index=range(len(info_dict))

    plan_list=[]
    for i in range(len(info_dict)):
        sfz=info_dict.iloc[i,0]
        plan_dir="../Multidimensional-time-series-with-transformer-main/0307data/"+sfz+"/tb_sub_plan.xlsx"
        plan=pd.read_excel(plan_dir)
        # print(plan.load[0])
        plan_list.append([len(plan),plan.load[0]])
    info_dict["plan"]=plan_list

    train_df=info_dict.iloc[:12,:]
    test_df=info_dict.iloc[12:,:]
    test_df.index=range(len(test_df))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    # train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    # test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_dataset=Clinical(train_df)
    test_dataset=Clinical(test_df)
    # print(test_dataset[0][0].shape)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    batchsize=4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    total_epoch = 1000
    train_num=range(1,total_epoch+1)
    avg_loss=[]
    for epoch in range(1, total_epoch+1):
        loss=train(optimizer,epoch,batchsize)
        avg_loss.append(loss)

    targetlist,predlist = test(epoch)
    print('target',targetlist)
    print('predict',predlist)
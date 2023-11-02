from model import Model,LMF_Model
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
from sklearn.utils import shuffle
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

import SimpleITK as sitk
from radiomics import featureextractor

import plotly.graph_objects as go
import plotly.io as pio
import kaleido
import seaborn as sns

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
        output = model(clinical_f.to(torch.float32),img_f.to(torch.float32))
        # print("output:",output,"plan:",wb_plan)
        criteria = nn.MSELoss()
        loss = criteria(output,wb_plan)
        # print(type(loss))
        # loss += fuse_loss 
        train_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # if batch_index % (3) == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]    Train Loss: {:.6f}'.format(
        #         epoch, batch_index, len(train_loader),
        #         100.0 * batch_index / len(train_loader), loss.item()/ bs))
    
    # if epoch%10==0: torch.save(model.state_dict(),'bone_resnet18.pth')
    if epoch%20==0:print('Train Epoch: {}   Train set: Average loss: {:.4f}'.format(
        epoch,train_loss/len(train_loader.dataset)))
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
            output = model(clinical_f.to(torch.float32),img_f.to(torch.float32))
            test_loss += criteria(output,wb_plan)
            targetcpu=wb_plan.cpu().numpy()
            predlist=np.append(predlist,output.cpu().numpy())

            targetlist=np.append(targetlist,targetcpu)
    return targetlist,predlist,test_loss

def split_dataset(data,k):
    jia,yi,bing=data[data.iloc[:,30]==0],data[data.iloc[:,30]==1],data[data.iloc[:,30]==2]
    jia=jia.reset_index(drop=True)
    yi=yi.reset_index(drop=True)
    bing=bing.reset_index(drop=True)
    score=[0]*k
    ta=[]
    pa=[]
    for i in range(k):
        df1=jia.drop(np.arange(int(np.floor(len(jia)*i/k)),int(np.floor(len(jia)*(i+1)/k))))
        df2=yi.drop(np.arange(int(np.floor(len(yi)*i/k)),int(np.floor(len(yi)*(i+1)/k))))
        df3=bing.drop(np.arange(int(np.floor(len(bing)*i/k)),int(np.floor(len(bing)*(i+1)/k))))
        df_train=pd.concat([df1,df2,df3],axis=0)
        df4=jia.iloc[int(np.floor(len(jia)*i/k)):int(np.floor(len(jia)*(i+1)/k)),:]
        df5=yi.iloc[int(np.floor(len(yi)*i/k)):int(np.floor(len(yi)*(i+1)/k)),:]
        df6=bing.iloc[int(np.floor(len(bing)*i/k)):int(np.floor(len(bing)*(i+1)/k)),:]
        df_test=pd.concat([df4,df5,df6],axis=0)
#         print(df_train)
#         print(df_test)
        X_train=df_train.iloc[:,0:30]
        Y_train=df_train.iloc[:,30]
        X_test=df_test.iloc[:,0:30]
        Y_test=df_test.iloc[:,30]
        clf = svm.SVC(C=2,kernel='rbf',tol=1,class_weight={0:1,1:0.9,2:1})
        clf.fit(X_train,Y_train)  # 训练分类器
        #print("Support Vector：\n", clf.n_support_)  # 每一类中属于支持向量的点数目
        #print(Y_test)
        #print("Predict：\n", clf.predict(X_test))  # 对测试集的预测结果
        score[i]=clf.score(X_test,Y_test)
        print(" Score[",i,"]=",clf.score(X_test,Y_test))
        #print(classification_report(Y_test,clf.predict(X_test)))
        r=np.array(Y_test)
        n=np.array(df_test.iloc[:,31])
        if i==0:
            ta=r.tolist()
            pa=clf.predict(X_test).tolist()
        else:
            ta.extend(r.tolist())
            pa.extend(clf.predict(X_test).tolist())
        #print(r)
        r=np.concatenate((r,n,clf.predict(X_test)),axis=0)
        cmp=DataFrame(r.reshape(3,int(r.shape[0]/3)))
        cmp.index=['labels','num','predict']
        pd.set_option('display.max_columns', None)
        print(cmp)
    print(score)
    score=np.average(score)
    print(score)
    return score,cmp,ta,pa

if __name__ == '__main__':
    model = ResNet18().cuda()
    model.load_state_dict(torch.load("bone_resnet18.pth"))

    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])

    transformer = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        normalize
    ])

    clinical_data = pd.read_excel("../lstm-gru-pytorch/clinical_with_plan3.xlsx")
    subplan = pd.read_excel("../lstm-gru-pytorch/tb_sub_plan.xlsx")
    for i in range(len(clinical_data)):
        if clinical_data.iloc[i,6]=='男':
            clinical_data.iloc[i,6]=1
        else:
            clinical_data.iloc[i,6]=0
    
    # clinical_data = clinical_data[clinical_data.iloc[:,6]==1]
    # clinical_data = clinical_data[clinical_data.iloc[:,4]/(clinical_data.iloc[:,5]/100)**2>=24]
    # clinical_data = clinical_data[clinical_data.iloc[:,3]<60]

    # clinical_data.年龄 = clinical_data.年龄/100
    # clinical_data.体重 = clinical_data.体重/100
    # clinical_data.身高 = clinical_data.身高/100
    # clinical_data.diagnose = clinical_data.diagnose/30
    # clinical_data.发病时间 = clinical_data.发病时间/360

    scaler = MinMaxScaler()

    clinical_data.年龄 = scaler.fit_transform(np.array(clinical_data.年龄).reshape(-1, 1))
    clinical_data.体重 = scaler.fit_transform(np.array(clinical_data.体重).reshape(-1, 1))
    clinical_data.身高 = scaler.fit_transform(np.array(clinical_data.身高).reshape(-1, 1))
    clinical_data.diagnose = scaler.fit_transform(np.array(clinical_data.diagnose).reshape(-1, 1))
    clinical_data.发病时间 = scaler.fit_transform(np.array(clinical_data.发病时间).reshape(-1, 1))

    xray_dir="../lstm-gru-pytorch/xray_with_plan/"
    clinical_list = []
    im_feat_list=[]
    plan_list = []
    for iii in range(len(clinical_data)):
        user_id = clinical_data.iloc[iii,0]
        body_weight = clinical_data.iloc[iii,4]*100
        sfz = clinical_data.iloc[iii,2]
        user_plan=subplan[subplan["user_id"]==user_id]
        latest_version=np.max(user_plan.version)
        xray=os.path.join(xray_dir,sfz)
        xray=os.path.join(xray,"post/b.JPG")
        if os.path.isfile(xray)!=1:
            xray = xray.replace(".JPG",".jpg")
        if os.path.isfile(xray)!=1:
            xray = xray.replace(".jpg",".png")

        '''
        # Transform input
        im_tiff = sitk.ReadImage(xray)
        im_vect = sitk.JoinSeries(im_tiff)  # Add 3rd dimension
        im = sitk.VectorImageSelectionCast(im_vect, 0, sitk.sitkFloat64)  # Select first color channel (if image is grayscale, all channels are equal)

        # Build full mask
        im_arr = sitk.GetArrayFromImage(im)
        ma_arr = np.ones(im_arr.shape)
        ma = sitk.GetImageFromArray(ma_arr)
        ma.CopyInformation(im)

        # Instantiate extractor
        extractor = featureextractor.RadiomicsFeaturesExtractor('path/to/config')  # optional supply parameter file)

        # extractor = featureextractor.RadiomicsFeaturesExtractor()  # Default settings 

        # Enable just first order
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')

        # Extract features
        results = extractor.execute(im, ma)
        print("!!!!!!!!!!!!!!!!!!!radiomics:",results)
        '''
        
        # print(xray)
        xray_img=Image.open(xray).convert('RGB')
        xray_img=transformer(xray_img)
        xray_img=xray_img.unsqueeze(0)
        model.eval()
        _,img_feature=model(xray_img.cuda())
        plan=user_plan[user_plan["version"]==latest_version].load.tolist()
        if len(plan)<=30:
            # plan = plan + [0]*(30-len(plan))
            # plan = plan/body_weight
            if len(plan)==0:
                continue
            im_feat_list.append(img_feature.cpu().detach().numpy()[0])
            clinical_list.append([clinical_data.iloc[iii,2],clinical_data.iloc[iii,3:].tolist()])
            plan_list.append([plan[0],len(plan)])
    
    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                        std=[0.33165374, 0.33165374, 0.33165374])

    transformer = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        normalize
    ])

    # linchuang_data=pd.read_excel("linchuang.xlsx")
    # for i in range(len(linchuang_data)):
    #     if linchuang_data.iloc[i,5]=='男':
    #         linchuang_data.iloc[i,5]=1
    #     else:
    #         linchuang_data.iloc[i,5]=0
    info_dict = clinical_list
    # for i in range(len(clinical_data)):
    #     info_dict.append()
    info_dict=DataFrame(info_dict)
    info_dict.columns=["sfz","clinical"]
    # im_feat_list=[]
    # for s in range(len(info_dict)):
    #     sfz=info_dict.loc[s,"sfz"]
    #     xray_dir="./0308xray/"+sfz+"/术后/"
    #     if not os.path.exists(xray_dir):
    #         xray_dir=xray_dir.replace("后","前")
    #     # print(xray_dir)
    #     xray=os.listdir(xray_dir)[0]
    #     xray=os.path.join(xray_dir,xray)
    #     xray_img=Image.open(xray)
    #     xray_img=transformer(xray_img)
    #     xray_img=xray_img.unsqueeze(0)
    #     model1.eval()
    #     _,img_feature=model1(xray_img.cuda())
    #     im_feat_list.append(img_feature.cpu().detach().numpy()[0])
    info_dict["img_feature"]=im_feat_list

    # drop_list=[]
    # for i in range(len(info_dict)):
    #     sfz=info_dict.iloc[i,0]
    #     plan_dir="../Multidimensional-time-series-with-transformer-main/0307data/"+sfz+"/tb_sub_plan.xlsx"
    #     if not os.path.exists(plan_dir):
    #         drop_list.append(i)
    # info_dict.drop(drop_list,axis=0,inplace=True)
    # info_dict.index=range(len(info_dict))

    # plan_list=[]
    # for i in range(len(info_dict)):
    #     sfz=info_dict.iloc[i,0]
    #     plan_dir="../Multidimensional-time-series-with-transformer-main/0307data/"+sfz+"/tb_sub_plan.xlsx"
    #     plan=pd.read_excel(plan_dir)
    #     # print(plan.load[0])
    #     plan_list.append([len(plan),plan.load[0]])
    info_dict["plan"]=plan_list

    mse_list = []
    
    k = 5


    cv_loss = 9999
    while (cv_loss>10):
        info_dict = shuffle(info_dict)
        target_weights = []
        predict_weights = []
        target_time = []
        predict_time = []
        for i in range(k):
            train_df=info_dict.drop(np.arange(int(np.floor(len(info_dict)*i/k)),int(np.floor(len(info_dict)*(i+1)/k))))
            test_df=info_dict.iloc[int(np.floor(len(info_dict)*i/k)):int(np.floor(len(info_dict)*(i+1)/k)),:]


        # train_df=info_dict.iloc[:85,:]
        # test_df=info_dict.iloc[85:,:]

            train_df.index=range(len(train_df))
            test_df.index=range(len(test_df))

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            batch_size = 16
            # train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
            # test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
            train_dataset=Clinical(train_df)
            test_dataset=Clinical(test_df)
            # print(test_dataset[0][0].shape)
            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)


            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # model = Model().to(device)
            model = LMF_Model().to(device)

            optimizer = optim.Adam(model.parameters(), lr=0.01)
            scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

            total_epoch = 300
            train_num=range(1,total_epoch+1)
            avg_loss=[]
            loss = 10
            epoch = 0
            for epoch in range(1, total_epoch+1):
                loss=train(optimizer,epoch,batch_size)
                avg_loss.append(loss)
                # scheduler.step()
            # while loss>0.5:
            #     epoch+=1
            #     loss=train(optimizer,epoch,batch_size)
            #     avg_loss.append(loss)

            targetlist,predlist,test_loss = test(epoch)
            predlist = np.round(predlist)
            print('target',targetlist)
            print('predict',predlist)
            print("test loss:",test_loss.cpu().numpy()/len(test_loader.dataset))
            mse_list.append(test_loss.cpu().numpy()/len(test_loader.dataset))
            # print(len(test_loader.dataset))


            tlen = len(targetlist)
            for n in range(int(tlen/2)):
                target_weights.append(targetlist[2*n])
                predict_weights.append(predlist[2*n])
                target_time.append(targetlist[2*n+1])
                predict_time.append(predlist[2*n+1])

        cv_loss = np.mean(mse_list)
        print("current cv loss:",cv_loss)
            
        #     x1= np.array([1,targetlist[2*n+1]])
        #     y1= np.array([targetlist[2*n],test_df.iloc[n,1][1]*100])
        #     xx1 = np.linspace(x1.min(), x1.max(), int(targetlist[2*n+1]))

        #     x2= np.array([1,predlist[2*n+1]])
        #     y2= np.array([predlist[2*n],test_df.iloc[n,1][1]*100])
        #     xx2 = np.linspace(x2.min(), x2.max(),int(predlist[2*n+1]))
        #     f1 = interp1d(x1, y1, kind = 'slinear')
        #     f2 = interp1d(x2, y2, kind = 'slinear')

        #     # fig = plt.figure(figsize=(8, 5))
        #     # plt.scatter(x1, y1)
        #     # plt.scatter(x2, y2)
        #     # # for n in ['slinear']:
        #     # plt.plot(xx1, f1(xx1),'-o', label= 'original plan')
        #     # plt.plot(xx2, f2(xx2),'-o', label= 'predicted plan')

        #     # plt.legend()
        #     # plt.ylabel(r"Weight", fontsize=10)
        #     # plt.xlabel(r"time(weeks)", fontsize=10)
        #     # plt.title("Plan prediction example")
        #     # plt.savefig("./prediction/fold-"+str(i+1)+'-'+str(n)+".png")
        #     # plt.show()

        #     fig = go.Figure()
        #     fig.add_trace(
        #         go.Scatter(
        #             name="Real plan",
        #             x=xx1,
        #             y=f1(xx1),
        #             mode="markers+lines",
        #             marker_size=8
        #     ))
        #     fig.add_trace(
        #         go.Scatter(
        #             name="Predicted plan",
        #             x=xx2,
        #             y=f2(xx2),
        #             mode="markers+lines",
        #             marker_size=8
        #     ))
            
        #     fig.update_layout(
        #         title="Plan prediction visualization",
        #         xaxis=dict(title="weeks", nticks=13),
        #         yaxis=dict(title="load (kg)", nticks=11, rangemode="tozero"),
        #         width=800,
        #         height=500
        #     )
        #     pio.write_image(fig,"./prediction/fold-"+str(i+1)+'-'+str(n)+".png",scale=5, width=800, height=500)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Real weights",
            x=np.array(range(len(target_weights))),
            y=target_weights,
            mode="markers+lines",
            marker_size=9
    ))
    fig.add_trace(
        go.Scatter(
            name="Predicted weights",
            x=np.array(range(len(predict_weights))),
            y=predict_weights,
            mode="markers+lines",
            marker_size=9
    ))
    
    fig.update_layout(
        title="Weights compare",
        xaxis=dict(title="patients", nticks=13),
        yaxis=dict(title="load (kg)", nticks=11, rangemode="tozero"),
        width=1000,
        height=500,
        # paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,1)'
    )
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    pio.write_image(fig,"./cmp/weights_comparison_auto.png",scale=5, width=1000, height=500)

    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter(
    #         name="Real time",
    #         x=np.array(range(len(target_time))),
    #         y=target_time,
    #         mode="markers+lines",
    #         marker_size=6
    # ))
    # fig.add_trace(
    #     go.Scatter(
    #         name="Predicted time",
    #         x=np.array(range(len(predict_time))),
    #         y=predict_time,
    #         mode="markers+lines",
    #         marker_size=6
    # ))
    
    # fig.update_layout(
    #     title="Time compare",
    #     xaxis=dict(title="patients", nticks=13),
    #     yaxis=dict(title="weeks", nticks=11, rangemode="tozero"),
    #     width=1000,
    #     height=500
    # )
    # pio.write_image(fig,"./cmp/time_comparison_auto.png",scale=5, width=1000, height=500)

    #Fig 2
    df1 = pd.DataFrame({'patient':np.array(range(len(target_time))), 
    'time': target_time,
    'weight':target_weights,
    "class":["target"]*len(target_time)})
    df2 = pd.DataFrame({'patient':np.array(range(len(predict_time))), 
    'time': predict_time,
    'weight':predict_weights,
    "class":["predict"]*len(predict_time)})
    df = pd.concat([df1,df2])

    # sns.set(rc={'figure.figsize':(30,8)})
                
    # ax = sns.barplot(y="time",x="patient",data=df,hue = "class")
    # plt.savefig("./cmp/time_comparison_auto.png", dpi=500)

    #Fig 3
    fig, ax = plt.subplots(figsize=(25,6))
    g = sns.lineplot(x="patient",  y="weight", data=df, ax=ax, hue = "class",
                marker='o', markersize=5, palette="Paired")
    ax2 = ax.twinx()
    g = sns.barplot(x="patient",  y="time", data=df, ax=ax2, hue="class",alpha=.5, palette="crest")
    # ax.set_title('All_comparison', fontsize = 17)
    ax.set(xticklabels=[],xlabel = 'patients')
    ax.set_ylabel("Weight (kg)")
    ax2.set_ylabel("Time (week)")
    ax.legend(title='Initial bearing weights',loc = "upper right")
    ax2.legend(title='Recovery time',loc = "upper left")
    plt.show()
    plt.savefig("./cmp/All_comparison_unimodal_img.png", dpi=500)

    print("CV loss:",np.mean(mse_list))
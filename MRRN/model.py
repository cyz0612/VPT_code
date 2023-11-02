from torch.nn import Module
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1032,32)
        self.fc2 = nn.Linear(128,2)
        self.autofuse = AutoFusion(135)

    def forward(self,info,img):
        
        batch_size=info.shape[0]
        # info=x[:,:,:,0]
        # img=x[:,:,:,1]

        if info.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        #张量外积
        # info_h = torch.cat((torch.ones(batch_size,1).type(DTYPE), info), dim=1)
        # img_h = torch.cat((torch.ones(batch_size,1).type(DTYPE), img), dim=1)
        
        # info_h=info_h.unsqueeze(2)
        # img_h=img_h.unsqueeze(1)

        # y=torch.bmm(info_h,img_h)
        # y = y.view(y.shape[0], -1)

        #AutoFusion
        # y = torch.cat((info,img),dim=1)

        # fuse_z = self.autofuse(y)
        # y = fuse_z['z']
        # fuse_loss = fuse_z['loss']

        y = img

        # y = self.fc1(y)
        # y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        return y

class AutoFusion(nn.Module):
    """docstring for AutoFusion"""
    def __init__(self, input_features):
        super(AutoFusion, self).__init__()

        self.input_features = input_features

        self.fuse_in = nn.Sequential(
            nn.Linear(input_features, input_features//2),
            nn.Tanh(),
            nn.Linear(input_features//2, 32),
            nn.ReLU()
            )
        self.fuse_out = nn.Sequential(
            nn.Linear(32, input_features//2),
            nn.ReLU(),
            nn.Linear(input_features//2, input_features)
            )
        self.criterion = nn.MSELoss()

    def forward(self, z):
        compressed_z = self.fuse_in(z)
        loss = self.criterion(self.fuse_out(compressed_z), z)
        output = {
            'z': compressed_z,
            'loss': loss
        }
        return output

class LMF_Model(Module):
    def __init__(self,rank=2):
        super(LMF_Model, self).__init__()

        self.info_hidden = 7
        self.img_hidden = 128


        self.output_dim = 128
        self.rank = rank
        self.use_softmax = 0

        # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.info_factor = Parameter(torch.Tensor(self.rank, self.info_hidden + 1, self.output_dim))
        self.img_factor = Parameter(torch.Tensor(self.rank, self.img_hidden + 1, self.output_dim))
    
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        xavier_normal(self.info_factor)
        xavier_normal(self.img_factor)
        xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128,32)
        self.fc2 = nn.Linear(32,2)
        self.autofuse = AutoFusion(135)

    def forward(self,info,img):
        
        batch_size=info.shape[0]
        # info=x[:,:,:,0]
        # img=x[:,:,:,1]

        if info.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        info_h = torch.cat((torch.ones(batch_size,1).type(DTYPE), info), dim=1)
        img_h = torch.cat((torch.ones(batch_size,1).type(DTYPE), img), dim=1)


        # _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        # _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)

        fusion_info = torch.matmul(info_h, self.info_factor)
        fusion_img = torch.matmul(img_h, self.img_factor)
        fusion_zy = fusion_info * fusion_img

        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        y = output.view(-1, self.output_dim)
        # if self.use_softmax:
        #     y = F.softmax(output)
        
        # y = y.view(batch_size, -1)

        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        return y
    
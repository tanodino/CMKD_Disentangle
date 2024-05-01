import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.autograd import Function
from torchvision.models import resnet18#, resnet50
import numpy as np


class FC_Classifier(torch.nn.Module):
    def __init__(self, n_classes):
        super(FC_Classifier, self).__init__()

        self.block = nn.Sequential(
            nn.LazyLinear(n_classes)
        )
    
    def forward(self, X):
        return self.block(X)

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * -ctx.alpha
        return output, None

def grad_reverse(x,alpha):
    return GradReverse.apply(x,alpha)


class ProjHead(torch.nn.Module):
    def __init__(self, out_dim):
        super(ProjHead, self).__init__()
        self.l1 = nn.LazyLinear(out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.l2 = nn.LazyLinear(out_dim)

    def forward(self,x):
        proj = self.l1(x)
        proj = self.bn1(proj)
        proj = F.relu(proj)        
        return  proj


class DisCoMKD(torch.nn.Module):
    def __init__(self, input_channel_first=4, input_channel_second=2, num_classes=10, f_encoder='image', s_encoder='image'):
        super(DisCoMKD, self).__init__()
        self.first_enc_inv = None
        self.second_enc_inv = None
        self.first_enc_spec = None
        self.second_enc_spec = None


        if f_encoder == 'image' or f_encoder == 'spectro' or f_encoder== 'thermal':
            first_enc_inv = resnet18(weights=None)
            first_enc_inv.conv1 = nn.Conv2d(input_channel_first, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.first_enc_inv = nn.Sequential(*list(first_enc_inv.children())[:-1])

            first_enc_spec = resnet18(weights=None)
            first_enc_spec.conv1 = nn.Conv2d(input_channel_first, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.first_enc_spec = nn.Sequential(*list(first_enc_spec.children())[:-1])
        elif f_encoder == 'mnist' :
            self.first_enc_inv = ModelEncoderLeNet()
            self.first_enc_spec = ModelEncoderLeNet()

        if s_encoder== 'image' or s_encoder == 'spectro' or s_encoder== 'thermal':
            second_enc_inv = resnet18(weights=None)
            second_enc_inv.conv1 = nn.Conv2d(input_channel_second, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.second_enc_inv = nn.Sequential(*list(second_enc_inv.children())[:-1])

            second_enc_spec = resnet18(weights=None)
            second_enc_spec.conv1 = nn.Conv2d(input_channel_second, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.second_enc_spec = nn.Sequential(*list(second_enc_spec.children())[:-1])
        elif s_encoder == "mnist":
            self.second_enc_inv = ModelEncoderLeNet()
            self.second_enc_spec = ModelEncoderLeNet()

        self.task_dom = FC_Classifier(2)
        self.task_dom2 = FC_Classifier(2)
        self.task_cl = nn.LazyLinear(num_classes)
        self.task_cl2 = nn.LazyLinear(num_classes)
        self.task_cl3 = FC_Classifier(num_classes)
        
        self.discr = FC_Classifier(2)

        self.projF = ProjHead(256)
        self.projS = ProjHead(256)

    def forward(self, x, lambda_val=1.):
        f_x, s_x = x
        f_emb_inv = self.first_enc_inv(f_x).squeeze()
        f_emb_spec = self.first_enc_spec(f_x).squeeze()
        s_emb_inv = self.second_enc_inv(s_x).squeeze()
        s_emb_spec = self.second_enc_spec(s_x).squeeze()
        
        nfeat = f_emb_inv.shape[1]//2
        
        f_shared_discr = self.projF(f_emb_inv)
        s_shared_discr = self.projS(s_emb_inv)

        f_domain_discr = f_emb_spec[:,0:nfeat]
        s_domain_discr = s_emb_spec[:,0:nfeat]
        f_domain_useless = f_emb_spec[:,nfeat::]
        s_domain_useless = s_emb_spec[:,nfeat::]

        f_task_feat = torch.cat([f_shared_discr,f_domain_discr],dim=1)
        s_task_feat = torch.cat([s_shared_discr,s_domain_discr],dim=1)

        pred_f_emb_dom = torch.cat( [self.task_dom(f_domain_discr), self.task_dom2(f_domain_useless)], dim=0)
        pred_s_emb_dom = torch.cat( [self.task_dom(s_domain_discr), self.task_dom2(s_domain_useless)], dim=0)

        return f_shared_discr, s_shared_discr, f_domain_discr, f_domain_useless, s_domain_discr, s_domain_useless, \
               pred_f_emb_dom, pred_s_emb_dom, \
               self.task_cl(f_task_feat), self.task_cl2(s_task_feat), \
               self.discr(grad_reverse(f_shared_discr,lambda_val)), self.discr(grad_reverse(s_shared_discr,lambda_val)), \
               self.task_cl3(f_shared_discr), self.task_cl3(s_shared_discr), \
               self.task_cl3(f_domain_discr),  self.task_cl3(s_domain_discr)

    def pred_firstEnc(self, x):        
        emb_inv = self.first_enc_inv(x).squeeze()
        nfeat = emb_inv.shape[1]//2
        emb_inv = self.projF(emb_inv)
        emb_spec = self.first_enc_spec(x).squeeze()
        task_feat = torch.cat([emb_inv,emb_spec[:,0:nfeat]],dim=1)
        return self.task_cl(task_feat)

    def pred_secondEnc(self, x):        
        emb_inv = self.second_enc_inv(x).squeeze()
        nfeat = emb_inv.shape[1]//2
        emb_inv = self.projS(emb_inv)
        emb_spec = self.second_enc_spec(x).squeeze()
        task_feat = torch.cat([emb_inv,emb_spec[:,0:nfeat]],dim=1)
        return self.task_cl2(task_feat)

class ModelEncoderLeNet(nn.Module):
    # network structure
    def __init__(self):
        super(ModelEncoderLeNet, self).__init__()
        self.conv1 = nn.LazyConv2d(6, 5, padding=2)
        self.conv2 = nn.LazyConv2d(16, 5)
        self.fc1 = nn.LazyLinear(512)
        #self.fc1   = nn.LazyLinear(120)
        #self.fc2   = nn.LazyLinear(84)
        #self.fc3   = nn.LazyLinear(10)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #print(x.shape)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        #print(x.shape)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        #x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        #print(x.shape)
        return x
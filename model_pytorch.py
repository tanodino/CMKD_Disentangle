import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.autograd import Function
from torchvision.models import resnet18#, resnet50
import numpy as np


class FC_Classifier(torch.nn.Module):
    def __init__(self, hidden_dims, n_classes, drop_probability=0.5):
        super(FC_Classifier, self).__init__()

        self.block = nn.Sequential(
            #nn.LazyLinear(hidden_dims),
            #nn.BatchNorm1d(hidden_dims),
            #nn.ReLU(),
            #nn.Dropout(p=drop_probability),
            nn.LazyLinear(n_classes)
        )
    
    def forward(self, X):
        return self.block(X)

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
        #print(alpha)
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
        #proj = F.relu(proj)
        #proj = self.bn1(proj)
        #proj = self.l2(proj)
        #gelu_z = F.gelu(proj)
        return  proj#F.gelu(proj) - F.gelu(proj).detach() + F.relu(proj).detach()#F.relu(proj) #proj
        #return proj



class CrossSourceModelGRLv2(torch.nn.Module):
    def __init__(self, input_channel_first=4, input_channel_second=2, num_classes=10, f_encoder='image', s_encoder='image'):
        super(CrossSourceModelGRLv2, self).__init__()
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
            
        #elif f_encoder == 'hyper':
        #    self.first_enc = ModelEncoderHyper(hidden_dims=1024)
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

        #elif s_encoder == 'hyper':
        #    self.second_enc = ModelEncoderHyper(hidden_dims=1024)
        elif s_encoder == "mnist":
            self.second_enc_inv = ModelEncoderLeNet()
            self.second_enc_spec = ModelEncoderLeNet()

        self.task_dom = nn.LazyLinear(2)
        self.task_cl = nn.LazyLinear(num_classes)
        self.discr = FC_Classifier(256, 2)

    def forward(self, x, lambda_val=1.):
        f_x, s_x = x
        #f_emb = self.first_enc(f_x).squeeze()
        #s_emb = self.second_enc(s_x).squeeze()
        #nfeat = f_emb.shape[1]
        f_emb_inv = self.first_enc_inv(f_x).squeeze()
        f_emb_spec = self.first_enc_spec(f_x).squeeze()
        s_emb_inv = self.second_enc_inv(s_x).squeeze()
        s_emb_spec = self.second_enc_spec(s_x).squeeze()
        return f_emb_inv, f_emb_spec, s_emb_inv, s_emb_spec, self.task_dom(f_emb_spec), self.task_dom(s_emb_spec), self.task_cl(f_emb_inv), self.task_cl(s_emb_inv), self.discr(grad_reverse(f_emb_inv,lambda_val)), self.discr(grad_reverse(s_emb_inv,lambda_val))

    def pred_firstEnc(self, x):        
        emb_inv = self.first_enc_inv(x).squeeze()
        return self.task_cl(emb_inv)

    def pred_secondEnc(self, x):        
        emb_inv = self.second_enc_inv(x).squeeze()
        return self.task_cl(emb_inv)







class CrossSourceModelGRL(torch.nn.Module):
    def __init__(self, input_channel_first=4, input_channel_second=2, num_classes=10, f_encoder='image', s_encoder='image'):
        super(CrossSourceModelGRL, self).__init__()
        self.first_enc = None
        self.second_enc = None

        if f_encoder == 'image' or f_encoder == 'spectro' or f_encoder== 'thermal':
            first_enc = resnet18(weights=None)
            first_enc.conv1 = nn.Conv2d(input_channel_first, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.first_enc = nn.Sequential(*list(first_enc.children())[:-1])
        elif f_encoder == 'hyper':
            self.first_enc = ModelEncoderHyper(hidden_dims=1024)
        elif f_encoder == 'mnist' :
            self.first_enc = ModelEncoderLeNet()

        if s_encoder== 'image' or s_encoder == 'spectro' or s_encoder== 'thermal':
            second_enc = resnet18(weights=None)
            second_enc.conv1 = nn.Conv2d(input_channel_second, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.second_enc = nn.Sequential(*list(second_enc.children())[:-1])
        elif s_encoder == 'hyper':
            self.second_enc = ModelEncoderHyper(hidden_dims=1024)
        elif s_encoder == "mnist":
            self.second_enc = ModelEncoderLeNet()

        self.task_dom = nn.LazyLinear(2)
        self.task_cl = nn.LazyLinear(num_classes)
        self.discr = FC_Classifier(256, 2)

    def forward(self, x, lambda_val=1.):
        f_x, s_x = x
        f_emb = self.first_enc(f_x).squeeze()
        s_emb = self.second_enc(s_x).squeeze()
        nfeat = f_emb.shape[1]
        f_emb_inv = f_emb[:,0:nfeat//2]
        f_emb_spec = f_emb[:,nfeat//2::]
        s_emb_inv = s_emb[:,0:nfeat//2]
        s_emb_spec = s_emb[:,nfeat//2::]
        return f_emb_inv, f_emb_spec, s_emb_inv, s_emb_spec, self.task_dom(f_emb_spec), self.task_dom(s_emb_spec), self.task_cl(f_emb_inv), self.task_cl(s_emb_inv), self.discr(grad_reverse(f_emb_inv,lambda_val)), self.discr(grad_reverse(s_emb_inv,lambda_val))

    def pred_firstEnc(self, x):        
        emb = self.first_enc(x).squeeze()
        nfeat = emb.shape[1]
        emb_inv = emb[:,0:nfeat//2]
        return self.task_cl(emb_inv)

    def pred_secondEnc(self, x):        
        emb = self.second_enc(x).squeeze()
        nfeat = emb.shape[1]
        emb_inv = emb[:,0:nfeat//2]
        return self.task_cl(emb_inv)





class CrossSourceModel(torch.nn.Module):
    def __init__(self, input_channel_first=4, input_channel_second=2, num_classes=10, f_encoder='image', s_encoder='image'):
        super(CrossSourceModel, self).__init__()
        self.first_enc = None
        self.second_enc = None

        if f_encoder == 'image' or f_encoder == 'spectro' or f_encoder== 'thermal':
            first_enc = resnet18(weights=None)
            first_enc.conv1 = nn.Conv2d(input_channel_first, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.first_enc = nn.Sequential(*list(first_enc.children())[:-1])
        elif f_encoder == 'hyper':
            self.first_enc = ModelEncoderHyper(hidden_dims=1024)
        elif f_encoder == 'mnist' :
            self.first_enc = ModelEncoderLeNet()

        if s_encoder== 'image' or s_encoder == 'spectro' or s_encoder== 'thermal':
            second_enc = resnet18(weights=None)
            second_enc.conv1 = nn.Conv2d(input_channel_second, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.second_enc = nn.Sequential(*list(second_enc.children())[:-1])
        elif s_encoder == 'hyper':
            self.second_enc = ModelEncoderHyper(hidden_dims=1024)
        elif s_encoder == "mnist":
            self.second_enc = ModelEncoderLeNet()

        self.task_dom = nn.LazyLinear(2)
        self.task_cl = nn.LazyLinear(num_classes)


    def forward(self, x):
        f_x, s_x = x
        f_emb = self.first_enc(f_x).squeeze()
        s_emb = self.second_enc(s_x).squeeze()
        nfeat = f_emb.shape[1]
        f_emb_inv = f_emb[:,0:nfeat//2]
        f_emb_spec = f_emb[:,nfeat//2::]
        s_emb_inv = s_emb[:,0:nfeat//2]
        s_emb_spec = s_emb[:,nfeat//2::]
        return f_emb_inv, f_emb_spec, s_emb_inv, s_emb_spec, self.task_dom(f_emb_spec), self.task_dom(s_emb_spec), self.task_cl(f_emb_inv), self.task_cl(s_emb_inv)

    def pred_firstEnc(self, x):        
        emb = self.first_enc(x).squeeze()
        nfeat = emb.shape[1]
        emb_inv = emb[:,0:nfeat//2]
        return self.task_cl(emb_inv)

    def pred_secondEnc(self, x):        
        emb = self.second_enc(x).squeeze()
        nfeat = emb.shape[1]
        emb_inv = emb[:,0:nfeat//2]
        return self.task_cl(emb_inv)


class MonoSourceModel(torch.nn.Module):
    def __init__(self, input_channel_first=4, encoder='image', num_classes=10):
        super(MonoSourceModel, self).__init__()
        
        self.first_enc = None
        if encoder == 'image' or encoder == 'spectro' or encoder== 'thermal':
            first_enc = resnet18(weights=None)
            first_enc.conv1 = nn.Conv2d(input_channel_first, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.first_enc = nn.Sequential(*list(first_enc.children())[:-1])
        elif encoder == 'hyper':
            self.first_enc = ModelEncoderHyper(hidden_dims=1024)
        elif encoder == 'mnist' :
            self.first_enc = ModelEncoderLeNet()

        self.task_cl = nn.LazyLinear(num_classes)
        
    def forward(self, x):
        emb = self.first_enc(x).squeeze()        
        pred = self.task_cl(emb)
        return pred



class ModelEncoderHyper(torch.nn.Module):
    def __init__(self, hidden_dims=1024, drop_probability=0.5):
        super(ModelEncoderHyper, self).__init__()
        self.block1 = nn.Sequential(
            nn.LazyLinear(hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

        self.block2 = nn.Sequential(
            nn.LazyLinear(hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, x):
        emb = self.block1(x)    
        emb = self.block2(emb)    
        return emb


class ModelEncoderMNIST(nn.Module):
    # network structure
    def __init__(self):
        super(ModelEncoderMNIST, self).__init__()
        self.conv1 = nn.LazyConv2d(32, 3, padding=1)
        self.conv2 = nn.LazyConv2d(64, 3, padding=1)
        self.fc1 = nn.LazyLinear(1024)

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
        #exit()
        x = F.relu(self.fc1(x))
        #print(x.shape)
        return x


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


class ModelHYPER(torch.nn.Module):
    def __init__(self, hidden_dims=512, drop_probability=0.5, num_classes=10):
        super(ModelHYPER, self).__init__()

        self.enc = ModelEncoderHyper()
        self.task_cl = nn.LazyLinear(num_classes)
        
    def forward(self, x):
        emb = self.enc(x)
        pred = self.task_cl(emb)
        return pred



class MultiSourceModel(torch.nn.Module):
    def __init__(self, input_channel_first=4, input_channel_second=2, num_classes=10, f_encoder='image', s_encoder='image', fusion_type='CONCAT'):
    #def __init__(self, input_channel_first=4, input_channel_second=2, num_classes=10, fusion_type='CONCAT'):
        super(MultiSourceModel, self).__init__()

        self.first_enc = None
        self.second_enc = None
        self.fusion_type  = fusion_type
        if f_encoder == 'image' or f_encoder == 'spectro' or f_encoder== 'thermal':
            first_enc = resnet18(weights=None)
            first_enc.conv1 = nn.Conv2d(input_channel_first, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.first_enc = nn.Sequential(*list(first_enc.children())[:-1])
        elif f_encoder == 'hyper':
            self.first_enc = ModelEncoderHyper(hidden_dims=1024)
        elif f_encoder == 'mnist' :
            self.first_enc = ModelEncoderLeNet()

        if s_encoder== 'image' or s_encoder == 'spectro' or s_encoder== 'thermal':
            second_enc = resnet18(weights=None)
            second_enc.conv1 = nn.Conv2d(input_channel_second, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.second_enc = nn.Sequential(*list(second_enc.children())[:-1])
        elif s_encoder == 'hyper':
            self.second_enc = ModelEncoderHyper(hidden_dims=1024)
        elif s_encoder == "mnist" :
            self.first_enc = ModelEncoderLeNet()
        
        
        self.task_cl = nn.LazyLinear(num_classes)
        
    def forward(self, x):
        f_x, s_x = x
        f_emb = self.first_enc(f_x).squeeze()
        s_emb = self.second_enc(s_x).squeeze()
        if self.fusion_type == 'CONCAT':
            emb = torch.cat([f_emb, s_emb],dim=1)
        else:
            emb = f_emb + s_emb
        
        pred = self.task_cl(emb)
        return pred

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, min_tau=.07, max_tau=1., t_period=50, eps=1e-7):
    #def __init__(self, temperature=1., min_tau=.07, max_tau=1., t_period=50, eps=1e-7):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.t_period = t_period
        self.eps = eps

    def forward(self, projections, targets, epoch=1):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")
        #temperature = self.min_tau + 0.5 * (self.max_tau - self.min_tau) * (1 + torch.cos(torch.tensor(torch.pi * epoch / self.t_period )))
        

        dot_product = torch.mm(projections, projections.T)
        ### For stability issues related to matrix multiplications
        #dot_product = torch.clamp(dot_product, -1+self.eps, 1-self.eps)
        ####GEODESIC SIMILARITY
        #print(projections)
        #print( dot_product )
        #print( torch.acos(dot_product) / torch.pi )
        #dot_product = 1. - ( torch.acos(dot_product) / torch.pi )

        dot_product_tempered = dot_product / self.temperature
        
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        stab_max, _ = torch.max(dot_product_tempered, dim=1, keepdim=True)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - stab_max.detach() ) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        #### FILTER OUT POSSIBLE NaN PROBLEMS #### 
        mdf = cardinality_per_samples!=0
        cardinality_per_samples = cardinality_per_samples[mdf]
        log_prob = log_prob[mdf]
        mask_combined = mask_combined[mdf]
        #### #### #### #### #### #### #### #### #### 

        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        return supervised_contrastive_loss
    

class SupervisedContrastiveLossV2(nn.Module):
    def __init__(self, temperature=0.07, min_tau=.07, max_tau=1., t_period=50, eps=1e-7):
    #def __init__(self, temperature=1., min_tau=.07, max_tau=1., t_period=50, eps=1e-7):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super(SupervisedContrastiveLossV2, self).__init__()
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.t_period = t_period
        self.eps = eps

    def forward(self, projections, targets, temperature=0.07):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")
        #temperature = self.min_tau + 0.5 * (self.max_tau - self.min_tau) * (1 + torch.cos(torch.tensor(torch.pi * epoch / self.t_period )))
        

        dot_product = torch.mm(projections, projections.T)
        ### For stability issues related to matrix multiplications
        #dot_product = torch.clamp(dot_product, -1+self.eps, 1-self.eps)
        ####GEODESIC SIMILARITY
        #print(projections)
        #print( dot_product )
        #print( torch.acos(dot_product) / torch.pi )
        #dot_product = 1. - ( torch.acos(dot_product) / torch.pi )

        dot_product_tempered = dot_product / temperature
        
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        stab_max, _ = torch.max(dot_product_tempered, dim=1, keepdim=True)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - stab_max.detach() ) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        #### FILTER OUT POSSIBLE NaN PROBLEMS #### 
        mdf = cardinality_per_samples!=0
        cardinality_per_samples = cardinality_per_samples[mdf]
        log_prob = log_prob[mdf]
        mask_combined = mask_combined[mdf]
        #### #### #### #### #### #### #### #### #### 

        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        return supervised_contrastive_loss

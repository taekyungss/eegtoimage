import numpy as np
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
# from pytorchtools import EarlyStopping
import time
from preprocess import get_data
import csv
# import sys
# sys.path.append('/home/chengxiangxin/mieeg') # 添加模块所在的文件夹路径
# import multi_head as mh




class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        # nn.utils.clip_grad_norm_(self.depthwise.parameters(), max_norm=1.0)  
        return out
    
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation = 1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        nn.init.kaiming_uniform_(self.conv1d.weight, nonlinearity='linear')

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv1d(x)
    

class TCN_block(nn.Module):
    def __init__(self, depth=2):
        super(TCN_block, self).__init__()
        self.depth = depth


        self.Activation_1 = nn.ELU()
        self.TCN_Residual_1 = nn.Sequential(
            #可能问题的所在
            CausalConv1d(32, 32, 4, dilation=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
            CausalConv1d(32, 32, 4, dilation=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
        )
        
        self.TCN_Residual = nn.ModuleList()
        self.Activation = nn.ModuleList()
        for i in range(depth-1):
            TCN_Residual_n = nn.Sequential(
            CausalConv1d(32, 32, 4, dilation=2**(i+1)),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
            CausalConv1d(32, 32, 4, dilation=2**(i+1)),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
        )
            self.TCN_Residual.append(TCN_Residual_n)
            self.Activation.append(nn.ELU())   
        
    def forward(self, x):
        block = self.TCN_Residual_1(x)
        # print(block.shape)
        block += x
        block = self.Activation_1(block)
        
        for i in range(self.depth-1):
            block_o = block
            block = self.TCN_Residual[i](block)
            block += block_o
            # block = torch.add(block_o,block)
            block = self.Activation[i](block)
        return block

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = input_size//(2*num_heads)
        self.num_heads = num_heads
        self.d_k = self.d_v = input_size // (num_heads * 2)
        
        self.W_Q = nn.Linear(input_size, self.hidden_size * num_heads)
        self.W_K = nn.Linear(input_size, self.hidden_size * num_heads)
        self.W_V = nn.Linear(input_size, self.hidden_size * num_heads)
        self.W_O = nn.Linear(self.hidden_size * num_heads, self.input_size)
        
        nn.init.normal_(self.W_Q.weight, mean=0.0, std=self.d_k ** -0.5)
        nn.init.normal_(self.W_K.weight, mean=0.0, std=self.d_k ** -0.5)
        nn.init.normal_(self.W_V.weight, mean=0.0, std=self.d_v ** -0.5)
        nn.init.normal_(self.W_O.weight, mean=0.0, std=self.d_v ** -0.5)

        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 计算Q、K、V
        Q = self.W_Q(x)   # (batch_size, seq_len, hidden_size * num_heads)
        K = self.W_K(x)   # (batch_size, seq_len, hidden_size * num_heads)
        V = self.W_V(x)   # (batch_size, seq_len, hidden_size * num_heads)
        # print(Q)

        # 将Q、K、V按头数进行切分
        Q = Q.view(batch_size, seq_len, self.num_heads, self.hidden_size).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, hidden_size)
        K = K.view(batch_size, seq_len, self.num_heads, self.hidden_size).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, hidden_size)
        V = V.view(batch_size, seq_len, self.num_heads, self.hidden_size).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, hidden_size)
        # print('切分',Q)
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.hidden_size ** 0.5)   # (batch_size, num_heads, seq_len, seq_len)
        attn_scores = attn_scores.softmax(dim=-1)

        attn_scores = self.dropout(attn_scores)
        # 计算注意力加权后的值
        attn_output = torch.matmul(attn_scores, V)   # (batch_size, num_heads, seq_len, hidden_size)
        # 将头拼接起来
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)   # (batch_size, seq_len, hidden_size * num_heads)
        # 计算输出
        output = self.W_O(attn_output)   # (batch_size, seq_len, hidden_size)
        return output, attn_scores


class attention_block(nn.Module):
    def __init__(self,):
        super(attention_block,self).__init__()
        self.LayerNorm = nn.LayerNorm(normalized_shape=32,eps=1e-06)
        # self.mha = nn.MultiheadAttention(32, 2,dropout=0.5, batch_first=True)
        self.mha = mh.MultiHeadAttention(2,32,0.5)
        # self.mha = MultiHeadAttention(32, 2)
        self.drop = nn.Dropout(0.2)
    
    def forward(self,x):
        #问题
        x = x.permute(2, 0, 1)
        # x = self.LayerNorm(x)
        # att_out,_ = self.mha(x,x,x)
        att_out = self.mha(query=x, key=x, value=x)
        att_out = self.drop(att_out)
        output = att_out.permute(1, 2, 0) + x.permute(1, 2, 0)
        return output

class conv_block(nn.Module):
    def __init__(self,):
        super(conv_block,self).__init__()
        self.conv_block_1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1,64), bias=False,padding='same'),
                nn.BatchNorm2d(16),
                # problem,
            )
        self.depthwise = nn.Conv2d(16, 16, (22,1), stride=1, padding=0, dilation=1, groups=16, bias=False)
        self.pointwise = nn.Conv2d(16, 16*2, 1, 1, 0, 1, 1, bias=False)
        self.conv_block_2 = nn.Sequential(
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.AvgPool2d(kernel_size=(1,8)),
                nn.Conv2d(32, 32, kernel_size=(1,16), bias=False,padding='same'),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 7)),
                nn.Dropout(0.5),
            )
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        out = self.conv_block_2(x)
        # nn.utils.clip_grad_norm_(self.depthwise.parameters(), max_norm=1.0)  
        return out

class ATCNet(nn.Module):

    def __init__(self, ):
        super(ATCNet, self).__init__()
        #conv模块
        # self.conv_block = nn.Sequential(
        #         nn.Conv2d(1, 16, kernel_size=(64,1), bias=False,padding='same'),
        #         nn.BatchNorm2d(16),
        #         # problem,
        #         DepthwiseConv2d(in_channels=16, out_channels=16*2, kernel_size=(1,22), stride=1, padding=0,dilation=1),
        #         nn.BatchNorm2d(32),
        #         nn.ELU(),
        #         nn.Dropout(0.3),
        #         nn.AvgPool2d(kernel_size=(8,1)),
        #         nn.Conv2d(32, 32, kernel_size=(16,1), bias=False,padding='same'),
        #         nn.BatchNorm2d(32),
        #         nn.ELU(),
        #         nn.AvgPool2d(kernel_size=(7, 1)),
        #         nn.Dropout(0.3),
        #     )
        # 没问题的模块
        # self.conv_block = nn.Sequential(
        #         nn.Conv2d(1, 16, kernel_size=(1,64), bias=False,padding='same'),
        #         nn.BatchNorm2d(16),
        #         # problem,
        #         DepthwiseConv2d(in_channels=16, out_channels=16*2, kernel_size=(22,1), stride=1, padding=0,dilation=1),
        #         nn.BatchNorm2d(32),
        #         nn.ELU(),
        #         nn.Dropout(0.5),
        #         nn.AvgPool2d(kernel_size=(1,8)),
        #         nn.Conv2d(32, 32, kernel_size=(1,16), bias=False,padding='same'),
        #         nn.BatchNorm2d(32),
        #         nn.ELU(),
        #         nn.AvgPool2d(kernel_size=(1, 7)),
        #         nn.Dropout(0.5),
        #     )
        self.conv_block = conv_block()
        #self-attention input_size,hidden_size,num_head
        self.attention_list = nn.ModuleList()
        self.TCN_list = nn.ModuleList()
        self.slideOut_list = nn.ModuleList()
        self.layerNorm_list = nn.ModuleList()
        for i in range(5):
            self.layerNorm_list.append(nn.LayerNorm(normalized_shape=32,eps=1e-06))
            self.attention_list.append(attention_block())
            self.TCN_list.append(TCN_block())
            self.slideOut_list.append(nn.Linear(32,4))

        # self.layerNormalization = nn.LayerNorm(normalized_shape=16,eps=1e-06 )
        # self.multihead_attn = attention_block()
        # self.TCN_block = TCN_block()
        # self.out_1 = nn.Linear(32,4)


        self.out_2 = nn.Linear(160,4)
        self.cv_out = nn.Linear(640,4)

    def forward(self, x):
        #64,1,22,1125
        # x = x.permute(0, 1, 3, 2)
        block1 = self.conv_block(x)
        #64,32,1,20
        # block1 = block1[:,:, -1,:]
        block1 = block1.squeeze(2)

        # block2 = self.multihead_attn(block1)
        # return 1
        # block2 = self.TCN_block(block2)

        fuse = 'average'
        n_windows = 5
        sw_concat = []
        for i in range(n_windows):
            # print(block1.shape)
            # print(i)
            st = i
            end = block1.shape[2]-n_windows+i+1 #在时间窗口上滑动
            # print(end)
            block2 = block1[:,:, st:end]  #获得时间窗口内的数据
            

            # block2 = self.layerNorm_list[i](block2.permute(0,2,1)).permute(0,2,1)

            # Attention_model
            # if attention is not None:
            # block2 = attention_block(block2) 
            block2 = self.attention_list[i](block2)

            # Temporal convolutional network (TCN)
            block3 = self.TCN_list[i](block2)
            # Get feature maps of the last sequence
            # 64,32,16
            block3 = block3[:,:, -1]
            # block3 = torch.functional.F.normalize(block3)
            
            # Outputs of sliding window: Average_after_dense or concatenate_then_dense
            if(fuse == 'average'):
                # block3 = block3.view(block3.size(0), -1)
                sw_concat.append(self.slideOut_list[i](block3))
            elif(fuse == 'concat'):
                if i == 0:
                    sw_concat = block3
                else:
                    sw_concat = torch.cat((sw_concat, block3), axis=1)

        if(fuse == 'average'):
            if len(sw_concat) > 1: # more than one window
                sw_concat = torch.stack(sw_concat).permute(1,0,2)
                # print(sw_concat[0])
                sw_concat = torch.mean(sw_concat, dim=1)
            else: # one window (# windows = 1)
                sw_concat = sw_concat[0]
        elif(fuse == 'concat'):
            sw_concat = self.out_2(sw_concat)

        # sw_concat = self.cv_out(block1.view(block1.size(0), -1))

        return sw_concat
 
## Take input of EEG and save it as a numpy array
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import config
from tqdm import tqdm
import numpy as np
import pdb
from natsort import natsorted
import cv2
from glob import glob
from torch.utils.data import DataLoader
from pytorch_metric_learning import miners, losses

import torch
import lpips
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import EEGDataset
from network import Conformer
# from model import ModifiedResNet
# from CLIPModel import CLIPModel
from torch.autograd import Variable
from visualizations import Umap, K_means, TsnePlot, save_image
from losses import ContrastiveLoss
from dataaugmentation import apply_augmentation

np.random.seed(45)
torch.manual_seed(45)

def train(epoch, model, optimizer, loss_fn, miner, train_data, train_dataloader, experiment_num):
        
    running_loss      = []
    eeg_featvec       = np.array([])
    eeg_featvec_proj  = np.array([])
    eeg_gamma         = np.array([])
    labels_array      = np.array([])

    tq = tqdm(train_dataloader)
    # for batch_idx, (eeg, eeg_x1, eeg_x2, gamma, images, labels) in enumerate(tq):
    for batch_idx, (eeg, images, labels) in enumerate(tq, start=1):
        # eeg_x1, eeg_x2 = eeg_x1.to(config.device), eeg_x2.to(config.device)
        eeg = torch.transpose(eeg, 1,2)
        eeg = torch.unsqueeze(eeg, dim=1)
        eeg = eeg.to(config.device).float()
        labels = labels.to(config.device).long()
        optimizer.zero_grad()
        # x1_proj, x1 = model(eeg_x1)
        # x2_proj, x2 = model(eeg_x2)
        # x_proj = model(eeg)
        outputs = model(eeg)

        hard_pairs = miner(outputs, labels)
        loss = loss_fn(outputs, labels, hard_pairs)
        
        # loss  = loss_fn(x1_proj, x2_proj)
        # backpropagate and update parameters
        loss.backward()
        optimizer.step()

        running_loss = running_loss + [loss.detach().cpu().numpy()]

        tq.set_description('Train:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

    if (epoch%config.vis_freq) == 0:
        # for batch_idx, (eeg, eeg_x1, eeg_x2, gamma, images, labels) in enumerate(tqdm(train_dataloader)):
        for batch_idx, (eeg, images, labels) in enumerate(tqdm(train_dataloader)):
            eeg = torch.transpose(eeg, 1,2)
            eeg = torch.unsqueeze(eeg, dim=1)
            eeg = eeg.to(config.device).float()
            labels = labels.to(config.device).long()
            with torch.no_grad():
                outputs = model(eeg)
            # eeg_featvec      = np.concatenate((eeg_featvec, x.cpu().detach().numpy()), axis=0) if eeg_featvec.size else x.cpu().detach().numpy()
            eeg_featvec_proj = np.concatenate((eeg_featvec_proj, outputs.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else outputs.cpu().detach().numpy()
            # eeg_gamma        = np.concatenate((eeg_gamma, gamma.cpu().detach().numpy()), axis=0) if eeg_gamma.size else gamma.cpu().detach().numpy()
            labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

        ### compute k-means score and Umap score on the text and image embeddings
        num_clusters   = 40
        # k_means        = K_means(n_clusters=num_clusters)
        # clustering_acc_feat = k_means.transform(eeg_featvec, labels_array)
        # print("[Epoch: {}, Train KMeans score Feat: {}]".format(epoch, clustering_acc_feat))

        k_means        = K_means(n_clusters=num_clusters)
        clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
        print("[Epoch: {}, Train KMeans score Proj: {}]".format(epoch, clustering_acc_proj))

        # tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
        # tsne_plot.plot(eeg_featvec, labels_array, clustering_acc_feat, 'train', experiment_num, epoch, proj_type='feat')
        

        tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
        tsne_plot.plot(eeg_featvec_proj, labels_array, clustering_acc_proj, 'train', experiment_num, epoch, proj_type='proj')

    return running_loss



if __name__ == '__main__':

    base_path       = config.base_path
    train_path      = config.train_path
    validation_path = config.validation_path
    device          = config.device

            
    #load the data
    ## Training data
    x_train_eeg = []
    x_train_image = []
    labels = []

    # ## hyperparameters
    batch_size     = config.batch_size
    EPOCHS         = config.epoch

    class_labels   = {}
    label_count    = 0

    for i in tqdm(natsorted(os.listdir(base_path + train_path))):
        loaded_array = np.load(base_path + train_path + i, allow_pickle=True)
        x_train_eeg.append(loaded_array[1].T)
        img = cv2.resize(loaded_array[0], (224, 224))
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))
        x_train_image.append(img)
        # if loaded_array[3] not in class_labels:
        # 	class_labels[loaded_array[3]] = label_count
        # 	label_count += 1
        # labels.append(class_labels[loaded_array[3]])
        labels.append(loaded_array[2])
        
    x_train_eeg   = np.array(x_train_eeg)
    x_train_image = np.array(x_train_image)
    train_labels  = np.array(labels)

    # ## convert numpy array to tensor
    x_train_eeg   = torch.from_numpy(x_train_eeg).float().to(device)
    x_train_image = torch.from_numpy(x_train_image).float().to(device)
    train_labels  = torch.from_numpy(train_labels).long().to(device)

    train_data       = EEGDataset(x_train_eeg, x_train_image, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True)


    ## Validation data
    x_val_eeg = []
    x_val_image = []
    label_Val = []

    for i in tqdm(natsorted(os.listdir(base_path + validation_path))):
        loaded_array = np.load(base_path + validation_path + i, allow_pickle=True)
        x_val_eeg.append(loaded_array[1].T)
        img = cv2.resize(loaded_array[0], (224, 224))
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))
        x_val_image.append(img)
        # if loaded_array[3] not in class_labels:
        # 	class_labels[loaded_array[3]] = label_count
        # 	label_count += 1
        # label_Val.append(class_labels[loaded_array[3]])
        label_Val.append(loaded_array[2])
        
    x_val_eeg   = np.array(x_val_eeg)
    x_val_image = np.array(x_val_image)
    val_labels  = np.array(label_Val)

    x_val_eeg   = torch.from_numpy(x_val_eeg).float().to(device)
    x_val_image = torch.from_numpy(x_val_image).float().to(device)
    val_labels  = torch.from_numpy(val_labels).long().to(device)

    val_data       = EEGDataset(x_val_eeg, x_val_image, val_labels)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True)

    model = 

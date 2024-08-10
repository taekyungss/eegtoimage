import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import config
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from pytorch_metric_learning import miners, losses
from network import Conformer
from visualizations import K_means, TsnePlot
from dataloader import EEGDataset
import cv2
from natsort import natsorted
from glob import glob

np.random.seed(45)
torch.manual_seed(45)

def train_stage1(epoch, model, optimizer, loss_fn, miner, train_dataloader, experiment_num):
    running_loss = []
    tq = tqdm(train_dataloader)

    for batch_idx, (eeg, images, labels) in enumerate(tq, start=1):
        eeg = torch.unsqueeze(torch.transpose(eeg, 1, 2), dim=1).to(config.device).float()
        labels = labels.to(config.device).long()

        optimizer.zero_grad()

        # Stage 1: Train with Triplet Loss
        eeg_embedding = model(eeg)
        hard_pairs = miner(eeg_embedding, labels)
        tri_loss = loss_fn(eeg_embedding, labels, hard_pairs)
        tri_loss.backward()
        optimizer.step()

        running_loss.append(tri_loss.detach().cpu().numpy())
        tq.set_description(f'Stage 1 Train: [{epoch}, {np.mean(running_loss):0.3f}]')

    return running_loss


def train_stage2(epoch, model, optimizer, train_dataloader, experiment_num):
    running_loss = []
    tq = tqdm(train_dataloader)

    # Load pre-trained ResNet50 for image encoding
    image_encoder = models.resnet50(pretrained=True).to(config.device)
    for param in image_encoder.parameters():
        param.requires_grad = False
    
    num_features = image_encoder.fc.in_features
    image_encoder.fc = nn.Sequential(
        nn.ReLU(),
        nn.Linear(num_features, config.eeg_channel, bias=False)
    ).to(config.device)

    for batch_idx, (eeg, images, labels) in enumerate(tq, start=1):
        eeg = torch.unsqueeze(torch.transpose(eeg, 1, 2), dim=1).to(config.device).float()
        images = images.to(config.device)

        optimizer.zero_grad()

        # Stage 2: Train with MSE Loss
        with torch.no_grad():
            image_embeddings = image_encoder(images)
        
        eeg_embedding = model(eeg)
        mse_loss = F.mse_loss(eeg_embedding, image_embeddings)
        mse_loss.backward()
        optimizer.step()

        running_loss.append(mse_loss.detach().cpu().numpy())
        tq.set_description(f'Stage 2 Train: [{epoch}, {np.mean(running_loss):0.3f}]')

    return running_loss


def validation_stage1(epoch, model, val_dataloader, experiment_num):
    running_loss_stage1 = []
    eeg_featvec_proj = np.array([])
    labels_array = np.array([])

    tq = tqdm(val_dataloader)

    # Define loss function and miner for Stage 1
    loss_fn_stage1 = losses.TripletMarginLoss(margin=0.2)
    miner = miners.MultiSimilarityMiner(epsilon=0.1)

    for batch_idx, (eeg, _, labels) in enumerate(tq, start=1):
        eeg = torch.unsqueeze(torch.transpose(eeg, 1, 2), dim=1).to(config.device).float()
        labels = labels.to(config.device).long()

        # Stage 1 Validation: Triplet Loss
        with torch.no_grad():
            eeg_embedding = model(eeg)
            hard_pairs = miner(eeg_embedding, labels)
            tri_loss = loss_fn_stage1(eeg_embedding, labels, hard_pairs)
        
        running_loss_stage1.append(tri_loss.detach().cpu().numpy())

        tq.set_description(f'Validation Stage 1: [{epoch}, Loss: {np.mean(running_loss_stage1):0.3f}]')

        # Update feature vectors and labels array
        eeg_featvec_proj = np.concatenate((eeg_featvec_proj, eeg_embedding.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else eeg_embedding.cpu().detach().numpy()
        labels_array = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

    # Compute k-means score on the projections
    num_clusters = 40
    k_means = K_means(n_clusters=num_clusters)
    clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
    print(f"[Epoch: {epoch}, Val Stage 1 KMeans score Proj: {clustering_acc_proj}]")

    tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
    tsne_plot.plot(eeg_featvec_proj, labels_array, clustering_acc_proj, 'val_stage1', experiment_num, epoch, proj_type='proj')

    return running_loss_stage1, clustering_acc_proj

def validation_stage2(epoch, model, val_dataloader, experiment_num):
    running_loss_stage2 = []
    eeg_featvec_proj = np.array([])
    labels_array = np.array([])

    tq = tqdm(val_dataloader)

    # Load pre-trained ResNet50 for image encoding
    image_encoder = models.resnet50(pretrained=True).to(config.device)
    for param in image_encoder.parameters():
        param.requires_grad = False
    
    num_features = image_encoder.fc.in_features
    image_encoder.fc = nn.Sequential(
        nn.ReLU(),
        nn.Linear(num_features, config.eeg_channel, bias=False)
    ).to(config.device)

    for batch_idx, (eeg, images, labels) in enumerate(tq, start=1):
        eeg = torch.unsqueeze(torch.transpose(eeg, 1, 2), dim=1).to(config.device).float()
        labels = labels.to(config.device).long()
        images = images.to(config.device)

        # Stage 2 Validation: MSE Loss
        with torch.no_grad():
            eeg_embedding = model(eeg)
            image_embeddings = image_encoder(images)
            mse_loss = F.mse_loss(eeg_embedding, image_embeddings)
        
        running_loss_stage2.append(mse_loss.detach().cpu().numpy())

        tq.set_description(f'Validation Stage 2: [{epoch}, Loss: {np.mean(running_loss_stage2):0.3f}]')

        # Update feature vectors and labels array
        eeg_featvec_proj = np.concatenate((eeg_featvec_proj, eeg_embedding.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else eeg_embedding.cpu().detach().numpy()
        labels_array = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

    # Compute k-means score on the projections
    num_clusters = 40
    k_means = K_means(n_clusters=num_clusters)
    clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
    print(f"[Epoch: {epoch}, Val Stage 2 KMeans score Proj: {clustering_acc_proj}]")

    tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
    tsne_plot.plot(eeg_featvec_proj, labels_array, clustering_acc_proj, 'val_stage2', experiment_num, epoch, proj_type='proj')

    return running_loss_stage2, clustering_acc_proj



if __name__ == '__main__':
    base_path       = config.base_path
    train_path      = config.train_path
    validation_path = config.validation_path
    device          = config.device

    # Load the data
    x_train_eeg, x_train_image, labels = [], [], []

    for i in tqdm(natsorted(os.listdir(base_path + train_path))):
        loaded_array = np.load(base_path + train_path + i, allow_pickle=True)
        x_train_eeg.append(loaded_array[1].T)
        img = cv2.resize(loaded_array[0], (224, 224))
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))
        x_train_image.append(img)
        labels.append(loaded_array[2])
        
    x_train_eeg   = np.array(x_train_eeg)
    x_train_image = np.array(x_train_image)
    train_labels  = np.array(labels)

    x_train_eeg   = torch.from_numpy(x_train_eeg).float().to(device)
    x_train_image = torch.from_numpy(x_train_image).float().to(device)
    train_labels  = torch.from_numpy(train_labels).long().to(device)

    train_data       = EEGDataset(x_train_eeg, x_train_image, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, pin_memory=False, drop_last=True)

    # Validation data
    x_val_eeg, x_val_image, label_Val = [], [], []

    for i in tqdm(natsorted(os.listdir(base_path + validation_path))):
        loaded_array = np.load(base_path + validation_path + i, allow_pickle=True)
        x_val_eeg.append(loaded_array[1].T)
        img = cv2.resize(loaded_array[0], (224, 224))
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))
        x_val_image.append(img)
        label_Val.append(loaded_array[2])
        
    x_val_eeg   = np.array(x_val_eeg)
    x_val_image = np.array(x_val_image)
    val_labels  = np.array(label_Val)

    x_val_eeg   = torch.from_numpy(x_val_eeg).float().to(device)
    x_val_image = torch.from_numpy(x_val_image).float().to(device)
    val_labels  = torch.from_numpy(val_labels).long().to(device)

    val_data       = EEGDataset(x_val_eeg, x_val_image, val_labels)
    val_dataloader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, pin_memory=False, drop_last=True)

    model     = Conformer().to(config.device)
    model     = torch.nn.DataParallel(model).to(config.device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=config.lr, betas=(0.9, 0.999))

    dir_info  = natsorted(glob('EXPERIMENT_*'))
    if len(dir_info)==0:
        experiment_num = 1
    else:
        experiment_num = int(dir_info[-1].split('_')[-1]) + 1

    if not os.path.isdir(f'EXPERIMENT_{experiment_num}'):
        os.makedirs(f'EXPERIMENT_{experiment_num}/val/tsne')
        os.makedirs(f'EXPERIMENT_{experiment_num}/train/tsne/')
        os.makedirs(f'EXPERIMENT_{experiment_num}/test/tsne/')
        os.makedirs(f'EXPERIMENT_{experiment_num}/test/umap/')
        os.makedirs(f'EXPERIMENT_{experiment_num}/finetune_ckpt/')
        os.makedirs(f'EXPERIMENT_{experiment_num}/finetune_bestckpt/')
        os.system(f'cp *.py EXPERIMENT_{experiment_num}')

    ckpt_lst = natsorted(glob(f'EXPERIMENT_{experiment_num}/finetune_ckpt/*'))
    print(f'EXPERIMENT:{experiment_num}')
    epoch = 0
    val_acc_lst = []

    # Train for multiple epochs with both stages
    for epoch in range(config.epochs):
        # Stage 1 training with Triplet Loss
        miner = miners.MultiSimilarityMiner(epsilon=0.1)
        loss_fn = losses.TripletMarginLoss(margin=0.2)
        train_loss_stage1 = train_stage1(epoch, model, optimizer, loss_fn, miner, train_dataloader, experiment_num)
        
        # Stage 2 training with MSE Loss
        train_loss_stage2 = train_stage2(epoch, model, optimizer, train_dataloader, experiment_num)
        
        # Validation for both stages
        val_loss_stage1, clustering_acc_proj_stage1 = validation_stage1(epoch, model, val_dataloader, experiment_num)
        val_loss_stage2, clustering_acc_proj_stage2 = validation_stage2(epoch, model, val_dataloader, experiment_num)

        val_acc_lst.append((clustering_acc_proj_stage1, clustering_acc_proj_stage2))

        # Save model checkpoints
        if (epoch % 5 == 0) or (epoch == config.epochs - 1):
            torch.save(model.state_dict(), f'EXPERIMENT_{experiment_num}/finetune_ckpt/epoch_{epoch}.pth')

        print(f"Epoch: {epoch}, Train Loss Stage 1: {np.mean(train_loss_stage1)}, Train Loss Stage 2: {np.mean(train_loss_stage2)}, Validation Loss Stage 1: {np.mean(val_loss_stage1)}, Validation Loss Stage 2: {np.mean(val_loss_stage2)}, Clustering Accuracy Stage 1: {clustering_acc_proj_stage1}, Clustering Accuracy Stage 2: {clustering_acc_proj_stage2}")

    # Save the final model
    torch.save(model.state_dict(), f'EXPERIMENT_{experiment_num}/finetune_bestckpt/epoch_{epoch}_final.pth')

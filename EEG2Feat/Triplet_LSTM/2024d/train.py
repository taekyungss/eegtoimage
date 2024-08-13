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
from torch.utils.data import DataLoader, Subset
from pytorch_metric_learning import miners, losses
import torch
import lpips
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import EEGImageNetDataset
from network import EEGFeatNet
# from model import ModifiedResNet
# from CLIPModel import CLIPModel
from visualizations import Umap, K_means, TsnePlot, save_image
from losses import ContrastiveLoss
from dataaugmentation import apply_augmentation
import time
import torchvision.transforms as transforms

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
    for batch_idx, (eeg,labels, image) in enumerate(tq, start=1):
        # eeg_x1, eeg_x2 = eeg_x1.to(config.device), eeg_x2.to(config.device)
        eeg    = eeg.to(config.device)
        labels = labels.to(config.device)

        optimizer.zero_grad()

        # x1_proj, x1 = model(eeg_x1)
        # x2_proj, x2 = model(eeg_x2)
        x_proj = model(eeg)

        hard_pairs = miner(x_proj, labels)
        loss       = loss_fn(x_proj, labels, hard_pairs)
        
        # loss  = loss_fn(x1_proj, x2_proj)
        # backpropagate and update parameters
        loss.backward()
        optimizer.step()

        running_loss = running_loss + [loss.detach().cpu().numpy()]

        tq.set_description('Train:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

    if (epoch%config.vis_freq) == 0:
        # for batch_idx, (eeg, eeg_x1, eeg_x2, gamma, images, labels) in enumerate(tqdm(train_dataloader)):
        for batch_idx, (eeg, labels, images) in enumerate(tqdm(train_dataloader)):
            eeg, labels = eeg.to(config.device), labels.to(config.device)
            with torch.no_grad():
                x_proj = model(eeg)
            # eeg_featvec      = np.concatenate((eeg_featvec, x.cpu().detach().numpy()), axis=0) if eeg_featvec.size else x.cpu().detach().numpy()
            eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
            # eeg_gamma        = np.concatenate((eeg_gamma, gamma.cpu().detach().numpy()), axis=0) if eeg_gamma.size else gamma.cpu().detach().numpy()
            labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

        ### compute k-means score and Umap score on the text and image embeddings
        num_clusters   = 67
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
 

def validation(epoch, model, optimizer, loss_fn, miner, train_data, val_dataloader, experiment_num):

	running_loss      = []
	eeg_featvec       = np.array([])
	eeg_featvec_proj  = np.array([])
	eeg_gamma         = np.array([])
	labels_array      = np.array([])

	tq = tqdm(val_dataloader)
	for batch_idx, (eeg, labels,images) in enumerate(tq, start=1):
		eeg, labels = eeg.to(config.device), labels.to(config.device)
		with torch.no_grad():
			x_proj = model(eeg)

			hard_pairs = miner(x_proj, labels)
			loss       = loss_fn(x_proj, labels, hard_pairs)

			running_loss = running_loss + [loss.detach().cpu().numpy()]

		tq.set_description('Val:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

		# eeg_featvec      = np.concatenate((eeg_featvec, x.cpu().detach().numpy()), axis=0) if eeg_featvec.size else x.cpu().detach().numpy()
		eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
		# eeg_gamma        = np.concatenate((eeg_gamma, gamma.cpu().detach().numpy()), axis=0) if eeg_gamma.size else gamma.cpu().detach().numpy()
		labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

	### compute k-means score and Umap score on the text and image embeddings
	num_clusters   = 67
	k_means        = K_means(n_clusters=num_clusters)
	clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
	print("[Epoch: {}, Val KMeans score Proj: {}]".format(epoch, clustering_acc_proj))

	tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
	tsne_plot.plot(eeg_featvec_proj, labels_array, clustering_acc_proj, 'val', experiment_num, epoch, proj_type='proj')

	return running_loss, clustering_acc_proj


class Args:
    dataset_dir = '/Data/summer24/data'
    subject = -1
    granularity = 'all'

    
if __name__ == '__main__':
    args = Args()
    start_time = time.time()
    device          = config.device

    # transform = transforms.Compose([
    #     transforms.Resize(256), 
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(), 
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # ])

    
    # ## hyperparameters
    batch_size     = config.batch_size
    EPOCHS         = config.epoch

    class_labels   = {}
    label_count    = 0

    dataset = EEGImageNetDataset(args, transform=None)
    print("Total dataset: ", len(dataset))
    train_index = np.array([i for i in range(len(dataset)) if i % 50 < 30])
    test_index = np.array([i for i in range(len(dataset)) if i % 50 > 29])

    train_subset = Subset(dataset, train_index)
    test_subset = Subset(dataset, test_index)
    print("train dataset : ",len(train_subset))
    print("test dataset : ",len(test_subset))


    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

    model     = EEGFeatNet(n_features=config.feat_dim, projection_dim=config.projection_dim, num_layers=config.num_layers).to(config.device)
    model     = torch.nn.DataParallel(model).to(config.device)
    optimizer = torch.optim.Adam(\
                                    list(model.parameters()),\
                                    lr=config.lr,\
                                    betas=(0.9, 0.999)
                                )

    
    dir_info  = natsorted(glob('EXPERIMENT_*'))
    if len(dir_info)==0:
        experiment_num = 1
    else:
        experiment_num = int(dir_info[-1].split('_')[-1]) + 1

    if not os.path.isdir('EXPERIMENT_{}'.format(experiment_num)):
        os.makedirs('EXPERIMENT_{}'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/val/tsne'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/train/tsne/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/test/tsne/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/test/umap/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/finetune_ckpt/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/finetune_bestckpt/'.format(experiment_num))
        os.system('cp *.py EXPERIMENT_{}'.format(experiment_num))

    ckpt_lst = natsorted(glob('EXPERIMENT_{}/checkpoints/eegfeat_*.pth'.format(experiment_num)))

    START_EPOCH = 0

    if len(ckpt_lst)>=1:
        ckpt_path  = ckpt_lst[-1]
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        START_EPOCH = checkpoint['epoch']
        print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
        START_EPOCH += 1
    else:
        os.makedirs('EXPERIMENT_{}/checkpoints/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/bestckpt/'.format(experiment_num))

    miner   = miners.MultiSimilarityMiner()
    loss_fn = losses.TripletMarginLoss()

    best_val_acc   = 0.0
    best_val_epoch = 0

    for epoch in range(START_EPOCH, EPOCHS):

        running_train_loss = train(epoch, model, optimizer, loss_fn, miner, train_subset, train_dataloader, experiment_num)
        if (epoch%config.vis_freq) == 0:
        	running_val_loss, val_acc   = validation(epoch, model, optimizer, loss_fn, miner, test_subset, val_dataloader, experiment_num)

        if best_val_acc < val_acc:
        	best_val_acc   = val_acc
        	best_val_epoch = epoch
        	torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
              }, 'EXPERIMENT_{}/bestckpt/eegfeat_{}_{}.pth'.format(experiment_num, 'all', val_acc))


        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
              }, 'EXPERIMENT_{}/checkpoints/eegfeat_{}.pth'.format(experiment_num, 'all'))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")



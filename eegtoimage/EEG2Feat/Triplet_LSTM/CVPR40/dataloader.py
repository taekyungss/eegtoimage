import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from natsort import natsorted
import os
import numpy as np
import config
import cv2
from dataaugmentation import apply_augmentation, extract_freq_band

# data를 로드하는 함수 정의 LoadData.load() -> x_eeg, x_image, labels, subjects 출력
class LoadData():
    def __init__(self, config, num_data=None, target_labels=None, target_subjects=None):
        self.base_path = config.base_path
        self.train_path = config.train_path
        self.validation_path = config.validation_path
        self.num_data = num_data
        self.target_labels = target_labels
        self.target_subjects = target_subjects
        
    def load_data(self):
        x_eeg = []
        x_image = []
        labels = []
        subjects = []

        file_list = os.listdir(self.base_path + self.train_path)
        
        # 설정값이 None이면 전체 데이터 다 불러오기 / 만약 None이 아니라 숫자면 해당 데이터 개수만큼 불러오기(미니 데이터셋으로 사용할때 필요)
        if self.num_data is not None:
            file_list = file_list[:self.num_data]

        for i in tqdm(file_list):
            loaded_array = np.load(self.base_path + self.train_path + i, allow_pickle=True)
            
            label = loaded_array[2]
            subject = loaded_array[4]
            # target label과 target subject가 설정되어 있으면 해당 숫자에 해당하는 label과 subject만 불러오기
            # 만일, None이면 전체 데이터 불러오기
            if (self.target_labels is None or label in self.target_labels) and (self.target_subjects is None or subject in self.target_subjects):
                x_eeg.append(loaded_array[1].T)
                img = cv2.resize(loaded_array[0], (224, 224))
                img = np.transpose(img, (2, 0, 1))
                x_image.append(img)
                labels.append(label)
                subjects.append(subject)

        x_eeg = np.array(x_eeg)
        x_image = np.array(x_image)
        labels = np.array(labels)
        subjects = np.array(subjects)

        return x_eeg, x_image, labels, subjects

class EEGDataset(Dataset):
    def __init__(self, eegs, images, labels, subjects):
        self.device = config.device
        self.eegs         =  torch.from_numpy(eegs).float().to(self.device)
        self.images       = torch.from_numpy(images).float().to(self.device)
        self.labels       = torch.from_numpy(labels).long().to(self.device)
        self.subjects     = torch.from_numpy(subjects).long().to(self.device)

        self.norm_max     = torch.max(self.eegs)
        self.norm_min     = torch.min(self.eegs)
        
    def __getitem__(self, index):
        eeg    = self.eegs[index]
        # dataset의 normalization은 최댓값 기준으로 -1,1로 scaling
        # eeg dataset(cvpr40) -> 음수존재
        norm   = torch.max(eeg) / 2.0
        eeg    = (eeg - norm)/ norm
        image  = self.images[index]
        label  = self.labels[index]
        subject = self.subjects[index]
        return eeg, image, label, subject

    def normalize_data(self, data):
        return (( data - self.norm_min ) / (self.norm_max - self.norm_min))

    def __len__(self):
        return len(self.eegs)

if __name__ == '__main__':
    target_labels = [1,29,30]  # 설정하지 않으면 None으로 설정
    target_subjects = [3,4]  # 설정하지 않으면 None으로 설정
    data_loader = LoadData(config, None, target_labels, target_subjects)  # num_data를 None으로 설정하여 전체 데이터를 불러옴
    x_train_eeg, x_train_image, labels, subjects = data_loader.load_data()
    train_data  = EEGDataset(x_train_eeg, x_train_image, labels, subjects)
    print(len(train_data))
    
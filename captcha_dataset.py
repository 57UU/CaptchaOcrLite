import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np

device="cuda:0" if torch.cuda.is_available() else "cpu"

class CaptchaDataset(Dataset):
    def __init__(self):
        self.x=torch.tensor(np.load("dataset/x.npy")/255.0).float().to(device)
        self.y=torch.tensor(np.load("dataset/y.npy")).long().to(device)
        # 忽略大小写
        self.y[self.y>=36]=self.y[self.y>=36]-26

    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)

captchaDataset = CaptchaDataset()

# 划分训练集和测试集，测试集占 10%
train_size = int(0.9 * len(captchaDataset))
test_size = len(captchaDataset) - train_size
_train_dataset, _test_dataset = torch.utils.data.random_split(captchaDataset, [train_size, test_size])

trainDataLoader = DataLoader(_train_dataset, batch_size=128, shuffle=True)
testDataLoader = DataLoader(_test_dataset, batch_size=512, shuffle=False)

class RealCaptchaDataset(Dataset):
    def __init__(self):
        self.x=torch.tensor(np.load("real_captcha/x.npy")/255.0).float().to(device)
        self.y=torch.tensor(np.load("real_captcha/y.npy")).long().to(device)
        # 忽略大小写
        self.y[self.y>=36]=self.y[self.y>=36]-26

    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)

realCaptchaDataset = RealCaptchaDataset()

test_size=100
_realCaptchaTrainDataset, _realCaptchaTestDataset = torch.utils.data.random_split(realCaptchaDataset, [test_size, len(realCaptchaDataset)-test_size])

realCaptchaTrainDataLoader = DataLoader(_realCaptchaTrainDataset, batch_size=16, shuffle=True)
realCaptchaTestDataLoader = DataLoader(_realCaptchaTestDataset, batch_size=512, shuffle=False)

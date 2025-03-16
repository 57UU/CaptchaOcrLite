from captcha_dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
import tqdm
import matplotlib.pyplot as plt
import numpy as np

num2char = {
    k:v for k,v in enumerate("0123456789abcdefghijklmnopqrstuvwxyz")
}
char2num = {
    k:v for v,k in enumerate("0123456789abcdefghijklmnopqrstuvwxyz")
}
char_length=len(num2char)

class CaptchaModel(nn.Module):
    def __init__(self):
        super(CaptchaModel, self).__init__()
        #cov
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        # 添加 BatchNorm2d 层
        self.bn1 = nn.BatchNorm2d(6) 
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        # 添加 BatchNorm2d 层
        self.bn2 = nn.BatchNorm2d(16) 
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*6*20, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.output_layer = nn.Linear(500, char_length*4)

    def forward(self, x):
        x = 1 - x # 反相
        # shape of x: (batch_size,26,80,3)
        x = x.permute(0, 3, 1, 2)
        # shape of x: (batch_size,3,26,80)
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        # shape of x: (batch_size,6,13,40)
        x = self.maxpool(F.relu(self.bn2(self.conv2(x))))
        # shape of x: (batch_size,16,6,20)
        x = x.flatten(1)
        # shape of x: (batch_size,16*6*20)
        x = F.relu(self.fc1(x))
        # shape of x: (batch_size,1000)
        x = F.relu(self.fc2(x))
        # shape of x: (batch_size,500)
        x = self.output_layer(x)
        # shape of x: (batch_size,char_length*4)
        return x

captchaModel = CaptchaModel().to(device)
optimizer = torch.optim.Adam(captchaModel.parameters(), lr=0.0002)
def train(epoch=2):
    captchaModel.train()
    
    for i in range(epoch):
        total_loss=0
        for x, y in tqdm.tqdm(trainDataLoader, desc=f'Epoch {i + 1}/{epoch}'):
            y_pred = captchaModel(x)
            optimizer.zero_grad()
            for j in range(4):
                loss = F.cross_entropy(y_pred[:, j * char_length:(j + 1) * char_length], y[:, j])
                total_loss+=loss.item()
                loss.backward(retain_graph=True)
            optimizer.step()
        print(f"Epoch {i + 1}/{epoch} loss: {total_loss / (len(trainDataLoader) * 4)}")

def test():
    captchaModel.eval()
    with torch.no_grad():
        single_correct = 0
        all_correct = 0
        total = 0
        for x, y in testDataLoader:
            y_pred = captchaModel(x)
            batch_size = x.size(0)
            statistics=torch.zeros(batch_size).to(device)
            for i in range(4):
                y_pred_i = y_pred[:, i * char_length:(i + 1) * char_length]
                y_i = y[:, i]
                y_pred_i = torch.argmax(y_pred_i, dim=1)
                # 计算单个字符的预测准确率
                statistics+=(y_pred_i == y_i)
            # 计算整个验证码的预测准确率
            single_correct += statistics.sum().item()
            all_correct += (statistics==4).sum().item()
            total += batch_size
        # 计算准确率
        single_accuracy = single_correct / (total * 4)
        all_accuracy = all_correct / total
        print(f"单个字符的准确率: {single_accuracy * 100:.3f}%")
        print(f"整个验证码的准确率: {all_accuracy * 100:.3f}%")
        return single_accuracy


def show_img_model(index=0):
    x,y=test_dataset[index]
    display_image(x)
    y_pred=predict(x)
    target_str="".join([num2char[i.item()] for i in y])
    print(f"预测结果:{y_pred}")
    print(f"真实结果:{target_str}")

def predict(x):
    captchaModel.eval()
    x.unsqueeze_(0)
    out=captchaModel(x)[0,:]
    result=[]
    # print(out.shape)
    for i in range(4):
        max_index=torch.argmax(out[i*char_length:(i+1)*char_length])
        result.append(num2char[max_index.item()])
    return "".join(result)

def display_image(x=train_dataset[0][0]):
    # 如果 x 是 PyTorch 张量，将其转换为 NumPy 数组
    if isinstance(x, torch.Tensor):
        x = x.cpu().float().numpy()
    # 如果图像数据是通道优先的格式 (C, H, W)，转换为通道最后 (H, W, C)
    if x.ndim == 3 and x.shape[0] in [1, 3]:
        x = np.transpose(x, (1, 2, 0))
    # 显示图像
    plt.imshow(x)
    plt.axis('off')
    plt.show()

def save_model(model=captchaModel):
    accuracy=test()
    torch.save(model.state_dict(), f"out/{accuracy * 100:.3f}.pth")

def load_model(model=captchaModel):
    import os
    files=[float(i.replace(".pth","")) for i in os.listdir("out")]
    files.sort(reverse=True)
    model_path=f"out/{files[0]}.pth"
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
    return model

load_model()

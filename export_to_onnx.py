import torch
import torch.nn as nn
import torch.nn.functional as F

num2char = {
    k:v for k,v in enumerate("0123456789abcdefghijklmnopqrstuvwxyz")
}
char_length = len(num2char)

class CaptchaModelLite(nn.Module):
    def __init__(self):
        super(CaptchaModelLite, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(5) 
        self.conv2 = nn.Conv2d(5, 7, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(7) 
        self.conv3 = nn.Conv2d(7, 10, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(10)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(10*3*10, 200)
        self.fc2 = nn.Linear(200, 150)
        self.output_layer = nn.Linear(150, char_length*4)

    def forward(self, x):
        x = 1 - x
        x = x.permute(0, 3, 1, 2)
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(F.relu(self.bn2(self.conv2(x))))
        x = self.maxpool(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    import os
    
    os.makedirs("export", exist_ok=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = CaptchaModelLite().to(device)
    model.load_state_dict(torch.load("out2/98.750.pth", map_location=device))
    model.eval()
    
    dummy_input = torch.randn(1, 26, 80, 3).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        "export/captcha_model.onnx",
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    print("ONNX model exported to export/captcha_model.onnx")

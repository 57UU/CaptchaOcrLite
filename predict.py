import torch
import numpy as np
from captcha_dataset import *
from model import *
import io
from PIL import Image


def predict_image_from_bytes(image:bytes):
    image=Image.open(io.BytesIO(image))
    return predict_image(image)

def predict_image(image:Image.Image)->str:
    print("size:",image.size)
    # 裁剪左上角26x80的部分
    image=image.crop((0,0,80,26))
    print("size:",image.size)
    image=image.convert("RGB")
    image=np.array(image)
    tensor=torch.tensor(image).float()/255.0
    print(tensor.shape)
    display_image(tensor)
    return predict(tensor.to(device))


def img_test(index=1):
    image=Image.open(f"real_captcha/img/{index}.png")
    print(predict_image(image))

if __name__=="__main__":
    img_test()
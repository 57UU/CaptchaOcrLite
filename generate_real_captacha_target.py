import ddddocr
import get_real_captcha
import json
from PIL import Image
import numpy as np

ocr=ddddocr.DdddOcr(beta=True)
ocr.set_ranges(6)#英文字母+数字

max_index=get_real_captcha.max_index

charset="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

char2num = {
    k:v for v,k in enumerate(charset)
}

x=[]
y=[]

def filter(target):
    if len(target)!=4:
        return False
    for i in target:
        if i not in charset:
            return False
    return True

for i in range(max_index):
    filename=f"real_captcha/img/{i}.png"
    try:
        img=Image.open(filename)
    except:
        print(f"file {filename} not found")
        continue
    target=ocr.classification(img)
    if not filter(target):
        continue
    x.append(np.array(img))
    y.append([char2num[j] for j in target])

    #print(f"filename: {filename}, target: {target}")

print("len(x):",len(x))
# save
np.save("real_captcha/x.npy",x)
np.save("real_captcha/y.npy",y)

import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter,ImageOps
import math
from matplotlib import pyplot as plt
import numpy as np

char2num = {
    k:v for v,k in enumerate("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
}

mono="fonts/JetBrainsMonoNL-Regular.ttf"
arial="fonts/ARIAL.TTF"

# 生成默认含4个字符验证码的图片
def check_code(width=80, height=26, char_length=4, font_file=arial, font_size=18):
    code = []
    img = Image.new(mode='RGB', size=(width, height), color=(255, 255, 255))
    img_txt = Image.new(mode='RGBA', size=(width, height), color=(255, 255, 255,0))
    draw = ImageDraw.Draw(img, mode='RGB')
    draw_txt = ImageDraw.Draw(img_txt, mode='RGBA')
    safe_margin = 13
    width_org=width
    width-=safe_margin*2

    def remove_o_O_mapper_cap(index:int):
        if index>=79:
            return index+1
        return index
    def remove_o_O_mapper_noncap(index:int):
        if index>=111:
            return index+1
        return index
    def rndChar():
        """
        生成随机大小写字母
        :return:
        """
        up_or_low = random.randint(0, 2)
        if up_or_low == 0:
            return chr(remove_o_O_mapper_noncap(random.randint(97, 121)))
        elif up_or_low == 1:
            return chr(remove_o_O_mapper_cap(random.randint(65, 89)))
        return chr(random.randint(49, 57)) #no 0
    
    def rndXOffset()->int:
        """
        生成随机偏移量
        :return:
        """
        return random.randint(-1,1)

    def rndColor():
        """
        生成随机颜色
        :return:
        """
        return (random.randint(0, 160), random.randint(0, 160), random.randint(0, 160))



    # # 写干扰点
    # for i in range(40):
    #     draw.point([random.randint(0, width_org), random.randint(0, height)], fill=rndColor())

    # # 写干扰圆圈
    # for i in range(40):
    #     draw.point([random.randint(0, width), random.randint(0, height)], fill=rndColor())
    #     x = random.randint(0, width)
    #     y = random.randint(0, height)
    #     draw.arc((x, y, x + 4, y + 4), 0, 90, fill=rndColor())

    # 画干扰线
    for i in range(15):
        x = random.randint(0, width_org)
        y = random.randint(0, height)
        length=random.randint(5,15)
        angle=random.random()*math.pi*2
        x_end=x+length*math.cos(angle)
        y_end=y+length*math.sin(angle)
        draw.line((x, y, x_end, y_end), fill="#6f6e70")

    # 写文字
    font = ImageFont.truetype(font_file, font_size)
    for i in range(char_length):
        char = rndChar()
        code.append(char2num[char])
        h = random.randint(-2, 2)
        draw_txt.text([i * width / char_length+rndXOffset()+safe_margin,h], char, font=font, fill=rndColor(),stroke_width=0.2)
    # 分离 alpha 通道
    r, g, b, alpha = img_txt.split()
    # 对 alpha 通道进行二值化处理（假设阈值为 128）
    alpha = alpha.point(lambda p: 255 if p > 128 else 0)
    # 重新组合图像
    img_txt.putalpha(alpha)

    # img = ImageOps.autocontrast(img)

    img = Image.alpha_composite(img.convert('RGBA'), img_txt)
    return img.convert("RGB"), code

is_test=True
is_test=False

if __name__ == '__main__':
    if is_test:
        img, code = check_code()
        #hide axis
        plt.axis('off')
        plt.imshow(img)
    else:
        x=[]
        y=[]
        for _ in range(100000):
            img, code = check_code()
            x.append(np.array(img))
            y.append(code)
        x=np.array(x)
        y=np.array(y)
        np.save("dataset/x.npy",x)
        np.save("dataset/y.npy",y)
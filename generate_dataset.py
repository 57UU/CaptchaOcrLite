import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math
import numpy as np

char2num = {
    k:v for v,k in enumerate("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
}

# 生成默认含4个字符验证码的图片
def check_code(width=80, height=26, char_length=4, font_file='JetBrainsMonoNL-Regular.ttf', font_size=20):
    code = []
    img = Image.new(mode='RGB', size=(width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, mode='RGB')
    safe_margin = 5

    width-=safe_margin*2

    def rndChar():
        """
        生成随机大小写字母
        :return:
        """
        up_or_low = random.randint(0, 2)
        if up_or_low == 0:
            return chr(random.randint(97, 122))
        elif up_or_low == 1:
            return chr(random.randint(65, 90))
        return chr(random.randint(48, 57))
    
    def rndXOffset()->int:
        """
        生成随机偏移量
        :return:
        """
        return random.randint(-3,3)

    def rndColor():
        """
        生成随机颜色
        :return:
        """
        return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

    # 写文字
    font = ImageFont.truetype(font_file, font_size)
    for i in range(char_length):
        char = rndChar()
        code.append(char2num[char])
        h = random.randint(-4, 4)
        draw.text([i * width / char_length+rndXOffset()+safe_margin,h], char, font=font, fill=rndColor())

    # 写干扰点
    for i in range(40):
        draw.point([random.randint(0, width), random.randint(0, height)], fill=rndColor())

    # # 写干扰圆圈
    # for i in range(40):
    #     draw.point([random.randint(0, width), random.randint(0, height)], fill=rndColor())
    #     x = random.randint(0, width)
    #     y = random.randint(0, height)
    #     draw.arc((x, y, x + 4, y + 4), 0, 90, fill=rndColor())

    # 画干扰线
    for i in range(8):
        x = random.randint(0, width)
        y = random.randint(0, height)
        length=random.randint(5,10)
        angle=random.random()*math.pi*2
        x_end=x+length*math.cos(angle)
        y_end=y+length*math.sin(angle)
        draw.line((x, y, x_end, y_end), fill="grey")


    # img = img.filter(ImageFilter.EDGE_ENHANCE)
    return img, code


if __name__ == '__main__':
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
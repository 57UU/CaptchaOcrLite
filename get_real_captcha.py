import time
import requests
import os
import base64


os.makedirs("real_captcha/img",exist_ok=True)

files=os.listdir("real_captcha/img")
nums=[int(file.split(".")[0]) for file in files]
nums.sort()
max_index=nums[-1]+1

def get_file_name():
    global max_index    
    name= f"real_captcha/img/{max_index}.png"
    max_index+=1
    return name

def get_url():
    # 生成当前时间戳并转换为毫秒
    timestamp = int(time.time() * 1000)
    # 替换URL中的时间戳
    return f"https://id.scu.edu.cn/api/public/bff/v1.2/one_time_login/captcha?_enterprise_id=scdx&timestamp={timestamp}"

def get_captcha()->bytes:
    url=get_url()
    headers={
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0" 
    }
    response=requests.get(url,headers=headers)
    data=response.json()
    base64_str:str=data["data"]["captcha"]
    bytes_data=base64.b64decode(base64_str)
    return bytes_data

def fetch_and_save_captcha():
    bytes_data=get_captcha()
    with open(get_file_name(),"wb") as f:
        f.write(bytes_data)
    return bytes_data

def get_numbers_of_captcha(num):
    for i in range(num):
        fetch_and_save_captcha()
        time.sleep(0.3)

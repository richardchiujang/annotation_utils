import os
import glob
import cv2
from PIL import Image, ImageOps

def check_max_size(input_path=None):
    input_path = r'C:\application\Develop\datasets\temp'
    os.chdir(input_path)
    images = glob.glob('*.jpg')
    Length = []
    Width = []
    for img in images:
        img = cv2.imread(img)
        width,length = img.shape[0:2]
        Length.append(length)
        Width.append(width)
    W = max(Width)
    L = max(Length)
    return [W,L]

# size=check_max_size()
# print(size)   # 2338 2338

def padding(img, expected_size):
    desired_size = expected_size
    # padding keep image aligned to the top left
    w,h = img.size
    # padding right
    img = ImageOps.pad(img, (desired_size,h), color='Gray', centering=(0, 0))
    # padding bottom 
    img = ImageOps.pad(img, (desired_size,desired_size), color='Gray', centering=(0, 0))
    return img

if __name__ == "__main__":

    # size=check_max_size()
    # print(size)   # 2338 2338

    expected_size = 2338
    input_path = r'C:\application\Develop\datasets\temp'
    os.chdir(input_path)
    images = glob.glob('*.jpg')   
    for img_path in images:
        img = Image.open(img_path)
        img = padding(img, expected_size)
        print(img.size) 
        img.save(img_path)

    


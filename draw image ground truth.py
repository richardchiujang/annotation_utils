"""
draw image ground truth
for DBNet.pytorch 繪製 dataset\train or test 的 ground truth 放在此 ./img_with_gt 裡面供檢查label正確性
base_path = 'C:/applications/datasets/'  # dataset location 
list_file = 'train.txt'                  # list file location  程式從這裡 train.txt or test.txt 讀取內容(清單)
program location C:\applications\annotation_tools\annotation_utils\
output location  C:\applications\annotation_tools\annotation_utils\img_with_gt
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob, shutil
import tqdm
import os

def _load_txt(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = [x.strip().strip('\ufeff').strip('\xef\xbb\xbf') for x in f.readlines()]
    return content

def show_img(imgs: np.ndarray, title='img'):
    color = (len(imgs.shape) == 3 and imgs.shape[-1] == 3)
    imgs = np.expand_dims(imgs, axis=0)
    for i, img in enumerate(imgs):
        plt.imsave('img_with_gt/'+title.split('/')[-1], img, cmap=None if color else 'gray')  # cmap='Greys')
#         plt.figure()
#         plt.title('{}_{}'.format(title, i))
#         plt.imshow(img, cmap=None if color else 'gray')
#     plt.show()
#     plt.savefig('./img_wgt/{}.jpg'.format(title))

def draw_bbox(img_path, result, color=(0, 0, 255), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
#         print(point.split(','))
        point = point.split(',')[:8]
        point = [int(x) for x in point]
        point = np.array(point)
        point = point.reshape((-1, 2))
        cv2.polylines(img_path, [point], isClosed=True, color=(255, 0, 0), thickness=3)
    return img_path


if __name__ == '__main__':    
    """
    draw image ground truth
    for DBNet.pytorch 
    繪製 (base_path) dataset\train or test 的 ground truth 放在此 ./img_with_gt 裡面供檢查label正確性
    base_path = 'C:/applications/datasets/'  # dataset location 
    list_file = 'train.txt'                  # list file location  程式從這裡 train.txt or test.txt 讀取內容(清單)
    program location C:\applications\annotation_tools\annotation_utils\
    output location  C:\applications\annotation_tools\annotation_utils\img_with_gt
    """

    try:
        os.mkdir('img_with_gt')
        print('create folder img_with_gt')
    except Exception as e:
        print(e)
        rmfiles = glob.glob('img_with_gt/*')
        for f in rmfiles:
            os.remove(f)
        print('clear files in img_with_gt folder ready.')

    base_path = r'C:/application\Develop/annotation_utils/preprocess_dataset/datasets_mix/datasets/' #'C:/application/Develop/datasets/'
    list_file = 'train.txt'
    f = open(base_path + list_file, 'r', encoding='utf-8')
    file_pair = f.readlines()
    f.close()
    for s in tqdm.tqdm(file_pair):
        a,b = s.rstrip('\n').split('\t')
        a,b = a.replace('./datasets/', ''), b.replace('./datasets/', '')
        content = _load_txt(base_path + b)
    #     print(a, b, len(content))
        if len(content) > 0:
            try:
                img = draw_bbox(base_path + a, content)
                show_img(img, a)
            except Exception as e:
                print(a, b, len(content), e)
        else:
            print(a, b, len(content))
"""
draw image ground truth
for DBNet.pytorch 繪製  
放在此 ./img_with_gt 裡面供檢查label正確性
base_path = 'procdataset'  
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

def show_img(imgs: np.ndarray, file_name):
    color = (len(imgs.shape) == 3 and imgs.shape[-1] == 3)
    imgs = np.expand_dims(imgs, axis=0)
    for i, img in enumerate(imgs):
        plt.imsave('img_with_gt/'+file_name, img, cmap=None if color else 'gray')  # cmap='Greys')
#         plt.figure()
#         plt.title('{}_{}'.format(title, i))
#         plt.imshow(img, cmap=None if color else 'gray')
#     plt.show()
#     plt.savefig('./img_wgt/{}.jpg'.format(title))

def draw_bbox(img_path, result, color=(0, 0, 255), thickness=2):
    if isinstance(img_path, str):
        # img = cv2.imread(img_path)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    draw_img = img.copy()
    for point in result:
#         print(point.split(','))
        point = point.split(',')[:8]
        point = [int(float(x)) for x in point]
        point = np.array(point)
        point = point.reshape((-1, 2))
        cv2.polylines(draw_img, [point], isClosed=True, color=(255, 0, 0), thickness=1)
    return draw_img


if __name__ == '__main__':    
    """
    draw image ground truth
    for DBNet.pytorch 
    先將資料集放在 procdataset 資料夾下 同時放入圖像與標籤檔案
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

    base_path = r'procdataset'   # 'procdataset' 
    file_list = os.listdir(base_path)
    n = 0
    for i in file_list:
        if ('.txt' in i):
            content = _load_txt(os.path.join(base_path, i))
            if len(content) > 0:
                try:
                    draw_img = draw_bbox(os.path.join(base_path, i.split('.txt')[0]+'.jpg'), content)
                    print('draw img {}'.format(i.split('.txt')[0]+'.jpg'))
                    show_img(draw_img, i.split('.txt')[0]+'.jpg')
                except Exception as e:
                    print('draw file {} occur error {}'.format(i, e))
            else:
                print(i, len(content))            

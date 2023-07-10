#  crop image using 4 pionts of possibly skewed rectangle.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob 

def get_bboxs(file_path):
    '''
    file_path: bboxs.txt with path
    only for model predict bboxs, can not use bboxs convert from xml
    bbox: 598,274,747,290,742,339,593,323,0.91943103
    return: bboxs list
    '''
    with open(file_path) as f:
        bbox_list = f.readlines()
        bboxs =[]
        for bbox in bbox_list:
            bbox = bbox.split(',')
            bbox = [int(x) for x in bbox[:8]] # get 4 points
            bboxs.append(bbox)
        # print(bboxs)
    return bboxs

def crop_bboxs(img, bboxs, save_name):
    '''
    5.定義4個源點（左上、右上、右下、左下）
    6.定義4個目標點（必須以與src點相同的順序列出！）
    7.使用cv2.getPerspectiveTransform（）來獲得M，即轉換矩陣
    8.使用cv2.warpPerspective（）來應用M並將圖像包為自上而下的視圖
    # 598,274,747,290,742,339,593,323,0.91943103
    # x1, y1, x2, y2, x3, y3, x4, y4, score = 598,274,747,290,742,339,593,323,0.91943103    
    '''
    # init variable
    i = 0
    dsts = ()
    h_count, v_count = 0, 0
    outbbox = []
    for bbox in bboxs:
        i += 1
        x1, y1, x2, y2, x3, y3, x4, y4 = bbox   # top left, top right, bottom right, bottom left
        horizon_length = max(x2-x1, x3-x4) # 水平長度
        verticle_length = max(y3-y2, y4-y1) # 垂直長度
        if horizon_length*1.1 > verticle_length: # 水平文字框，長邊為水平邊。如果接近正方形(*1.1)，則長邊為水平邊
            h_count += 1
            vertical_shift = 0.05 # 左右邊界擴展比例
            left_horizon_shift = 0.05 # 上邊界擴展比例
            right_horizon_shift = 0.25 # 下邊界擴展比例             
            w_box = int(horizon_length + horizon_length * vertical_shift) # 長邊
            h_box = int(verticle_length + verticle_length * (left_horizon_shift + right_horizon_shift)) # 文字高度
            word_height = 32 # 輸出文字高度
            shrink_rate = h_box/word_height  # 圖形縮放比例         
            pts1 = np.float32([[int(x1-w_box*vertical_shift), int(y1-h_box*left_horizon_shift)],[int(x2+w_box*vertical_shift), int(y2-h_box*left_horizon_shift)],[int(x3+w_box*vertical_shift), int(y3+h_box*right_horizon_shift)],[int(x4-w_box*vertical_shift), int(y4+h_box*right_horizon_shift)]])
            pts2 = np.float32([[0,0],[w_box/shrink_rate,0],[w_box/shrink_rate,h_box/shrink_rate],[0,h_box/shrink_rate]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(img,M,(int(w_box/shrink_rate),int(h_box/shrink_rate)))
            outbbox.append([x1, y1, x2, y2, x3, y3, x4, y4, 'horizon'])
            cv2.imencode('.jpg', dst)[1].tofile('procdataset/crop/{}_crop_h_{}.jpg'.format(save_name, str(i)))
        else: # 垂直文字框，長邊為垂直邊，逆時針旋轉90度
            v_count += 1
            horizon_shift = 0.05 # 上下邊界擴展比例
            left_vertical_shift = 0.1 # 左邊界擴展比例
            right_vertical_shift = 0.25 # 右邊界擴展比例             
            w_box = int(verticle_length + verticle_length * horizon_shift) # 長邊(直邊)
            h_box = int(horizon_length + horizon_length * (left_vertical_shift + right_vertical_shift)) # 文字寬度
            word_height = 32 # 輸出文字寬度
            shrink_rate = h_box/word_height  # 圖形縮放比例
            pts1 = np.float32([[int(x1-h_box*left_vertical_shift), int(y1-w_box*horizon_shift)],[int(x2+h_box*right_vertical_shift), int(y2-w_box*horizon_shift)],[int(x3+h_box*right_vertical_shift), int(y3+w_box*horizon_shift)],[int(x4-h_box*left_vertical_shift), int(y4+w_box*horizon_shift)]])
            pts2 = np.float32([[0,0],[h_box/shrink_rate,0],[h_box/shrink_rate,w_box/shrink_rate],[0,w_box/shrink_rate]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(img,M,(int(h_box/shrink_rate),int(w_box/shrink_rate)))
            dst = cv2.rotate(dst, cv2.ROTATE_90_COUNTERCLOCKWISE) # 逆時針旋轉90度
            outbbox.append([x1, y1, x2, y2, x3, y3, x4, y4, 'vertical'])
            cv2.imencode('.jpg', dst)[1].tofile('procdataset/crop/{}_crop_v_{}.jpg'.format(save_name, str(i)))
        
        # cv2.imwrite(, dst)
        
        dsts = dsts.__add__((dst,))
        # plt.subplot(121),plt.imshow(img),plt.title('Input')
        # plt.subplot(122),plt.imshow(dst),plt.title('Output')
        # plt.show()

    # print(outbbox)
    if h_count > v_count:
        return 'horizontal', len(dsts)
    else:
        return 'vertical', len(dsts)

if __name__ == '__main__':
    file_list = glob.glob('procdataset/*.txt')
    # print(file_list)
    for file_path in file_list:
        # print(file_path)
        bboxs = get_bboxs(file_path)
        img = cv2.imdecode(np.fromfile(file_path.replace('.txt', '.jpg'), dtype=np.uint8), -1)
        save_name = file_path.split('\\')[-1].split('.')[0]
        print(crop_bboxs(img, bboxs, save_name), file_path)

# img = cv2.imread()
# img = cv2.imdecode(np.fromfile('procdataset/Wealth財訊674台積電再造日本矽島202212_300_040.jpg', dtype=np.uint8), -1)
# rows,cols,ch = img.shape
# print(rows,cols,ch)
# bboxs = get_bboxs(file_path)
# file_path='procdataset/Wealth財訊674台積電再造日本矽島202212_300_040.txt'
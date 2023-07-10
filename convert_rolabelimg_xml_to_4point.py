# 這不用了，robndbox 位置會一點點跑掉

#! python3
# -*- encoding: utf-8 -*-
'''
convert rolabelimg xml (will process both robndbox and bndbox) to dbnet (x1, y1, x2, y2, x3, y3, x4, y4, class_name) txt
xml with robndbox need image size in xml content
這會跑一整個資料夾內的 xml 檔案

@File    :   convert_rolabelimg_xml_to_4point_txt.py
@Time    :   2022/12/14 
@Author  :   richard
@Version :   1.0
@Contact :   rchiu3210@gmail.com
@License :   MIT LICENSE
@Desc    :   <awaiting description>
'''

from logging import root
import math
import xml.etree.ElementTree as ET
import os
import glob
import sys
import numpy as np
from pathlib import Path

def convert_rolabelimg2_4point(xml_path:str) -> None:
    """
    Args: 
        - `xml_path` (str) : path to roLabelImg label file, like /xx/xx.xml
        
    Returns: 
        - `box_points` (list): shape (N, 8 + 1), N is the number of objects, 8 + 1 is \
            `(x1, y1, x2, y2, x3, y3, x4, y4, class_name)`
    """
    
    """
    for robndbox to 4 point 
    <robndbox>
      <cx>654.0</cx>
      <cy>1085.0</cy>
      <w>6.0</w>
      <h>19.0</h>
      <angle>0.0</angle>
    </robndbox>
    """    
    with open(xml_path) as f:
        tree = ET.parse(f)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        objects = root.iter('object')
        boxes = [] # list of tuple(cz, cy, w, h, angle), angle is in [0-pi)
        for obj in objects:
            if obj.find('type').text == 'robndbox':
                rbox_node = obj.find('robndbox')
                cat = obj.find('name').text
                rbox = dict()
                for key in ['cx', 'cy', 'w', 'h', 'angle']:
                    rbox[key] = float(rbox_node.find(key).text)
                boxes.append(list((*rbox.values(), cat)))
        # print(f"bboxes: {boxes}")
        
        # https://blog.csdn.net/lx_ros/article/details/126093044  
        box_points = [] # list of box defined with four vertices
        for box in boxes:
            cx, cy, w, h, ag, cat = box
            alpha_w = math.atan(w / h)
            alpha_h = math.atan(h / w)
            d = math.sqrt(w**2 + h**2) / 2 # 對角線長度
            if ag > math.pi / 2:   # 90
                beta = ag - math.pi / 2 + alpha_w
                if beta <= math.pi / 2:
                    x1, y1 = cx + d * math.cos(beta), cy + d * math.sin(beta)
                    x2, y2 = cx - d * math.cos(beta), cy - d * math.sin(beta)
                elif beta > math.pi / 2:
                    beta = math.pi - beta
                    x1, y1 = cx - d * math.cos(beta), cy + d * math.sin(beta)
                    x2, y2 = cx + d * math.cos(beta), cy - d * math.sin(beta)
                x3, y3 = x1 - h * math.cos(ag - math.pi / 2), y1 - h * math.sin(ag - math.pi / 2)
                x4, y4 = x2 + h * math.cos(ag - math.pi / 2), y2 + h * math.sin(ag - math.pi / 2) 
            elif ag <= math.pi / 2:  # 90
                beta = ag + alpha_h
                if beta <= math.pi / 2:
                    x1, y1 = cx + d * math.cos(beta), cy + d * math.sin(beta)
                    x2, y2 = cx - d * math.cos(beta), cy - d * math.sin(beta)
                elif beta > math.pi / 2:
                    beta = math.pi - beta
                    x1, y1 = cx - d * math.cos(beta), cy + d * math.sin(beta)
                    x2, y2 = cx + d * math.cos(beta), cy - d * math.sin(beta)
                x3, y3 = x1 - w * math.cos(ag), y1 - w * math.sin(ag)
                x4, y4 = x2 + w * math.cos(ag), y2 + w * math.sin(ag)
            points = np.array([x1, y1, x3, y3, x2, y2, x4, y4], dtype=np.int32)  # np.array
            points[0::2] = np.clip(points[0::2], 0, width)  # 從0開始每兩個取一個值 = 1,3,5,7 四點 x1,x3,x2,x4 的 min(0), max(width)
            points[1::2] = np.clip(points[1::2], 0, height) # 從1開始每兩個取一個值 = 2,4,6,8 四點 y1,y3,y2,y4 的 min(0), max(height)
            # 因為 4點的順序與預期不同，所以新增調整順序
            points = np.concatenate((points[:2], points[4:6], points[2:4], points[6:]),axis=0)  # [x1, y1, x3, y3, x2, y2, x4, y4] -> [x1, y1, x2, y2, x3, y3, x4, y4]
            box_points.append([*points, cat])
        # print(box_points)

    # for bndbox to 4 point
    """
    <bndbox>
        <xmin>1009</xmin>
        <ymin>86</ymin>
        <xmax>1051</xmax>
        <ymax>380</ymax>
    </bndbox>
    """
    with open(xml_path) as f:    
        tree = ET.parse(f)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        objects = root.iter('object')
        boxes = [] # list of tuple(xmin, ymin, xmax, ymax) -> xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax -> 0,1,2,1,2,3,0,3
        for obj in objects:
            if obj.find('type').text == 'bndbox':
                box_node = obj.find('bndbox')
                cat = obj.find('name').text
                box = dict()
                for key in ['xmin', 'ymin', 'xmax', 'ymax']:
                    box[key] = float(box_node.find(key).text)
                boxes.append(list((*box.values(), cat)))
        for box in boxes:
            x1,y1,x2,y2,x3,y3,x4,y4 = int(box[0]),int(box[1]),int(box[2]),int(box[1]),int(box[2]),int(box[3]),int(box[0]),int(box[3])
            box_points.append([x1,y1,x2,y2,x3,y3,x4,y4, cat])
        print(len(box_points))

                            
    return box_points
    

def roLabelImg2DOTA(xml_dir):
    """ convert roLabelImg xml format (cx,cy,w,h,angle) annotation to DOTA Dataset text format \
        (x1, y1, x2, y2, x3, y3, x4, y4, class_name)
    
    Args: 
        - xml_dir (str): path to roLabelImg xml annotation files, like `data/xmls`     
    """
    p = os.path.dirname(xml_dir)
    p = Path(p) / "labels"
    p.mkdir(parents=True, exist_ok=True)
    p = str(p)
    xmls = glob.glob(os.path.join(xml_dir, '*.xml'))
    for name in xmls:
        boxes = convert_rolabelimg2_4point(name)
        base_name = os.path.splitext(os.path.basename(name))[0]
        with open(os.path.join(p, f"{base_name}.txt"), 'w') as f:
            for box in boxes:
                f.write(f"{', '.join(list(map(lambda x: str(x), box)))}\n")
            
# def test(xml_path, img_path):
#     import cv2
#     xml_dir = xml_path
#     base_name = os.path.splitext(os.path.basename(xml_dir))[0]
# #     img_path = os.path.join(os.path.dirname(xml_dir), "../", "images",  f"{base_name}.png")
#     img_path = 'demo/demo2.png'
#     img = cv2.imread(img_path)
#     height, width, _ = img.shape
#     boxes = convert_rolabelimg2dota(xml_dir)
#     contours = [] 
#     for box in boxes:
#         contours.append(box[:8])
#     boxes = np.array(contours, dtype=np.int32)
#     contours = boxes.reshape(-1, 4, 2)
#     for i in range(len(contours)):
#         cv2.drawContours(img, contours, i, (0, 255, 0), 3)
#     cv2.imwrite(xml_dir.replace(".xml", "_n.png"), img)
    
    
if __name__ == '__main__':
    files_path = 'procdataset'
    print(files_path)
    # txt_file = 'test/presentation_032.txt'
    roLabelImg2DOTA(files_path)
    print('process {} to labels folder'.format(files_path))




    # test()
    # xml_path = '先探週刊第222期拜登訪台積電信賴產業鏈值千金2022-12-08i_001.xml'
    # img_path = '先探週刊第222期拜登訪台積電信賴產業鏈值千金2022-12-08i_001.jpg'
    # roLabelImg2DOTA(xml_path)
    # convert_rolabelimg2_4point(xml_path)
    # roLabelImg2DOTA('./')
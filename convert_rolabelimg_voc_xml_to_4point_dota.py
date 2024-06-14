#! python3
# -*- encoding: utf-8 -*-
'''
convert rolabelimg xml (will process both robndbox and bndbox) to dbnet (x1, y1, x2, y2, x3, y3, x4, y4, class_name) txt
rolabelimg xml 為 VOC 格式 + ','
dbnet 4 points 為 DOTA 格式
xml with robndbox need image size in xml content
這會跑一整個資料夾內的 xml 檔案

@File    :   convert_rolabelimg_voc_xml_to_4point_dota.py
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

def sort_points_clockwise(PointList):
    """提供一组凸四边形顶点，按照从左上角顶点开始顺时针方向排序
      :param points: Numpy矩阵，shape为(4, 2)，描述一组凸四边形顶点
      :return sorted_points: 经过排序的点
    """
    # 直接分别取四个点坐标中x和y的最小值作为最大外接矩形左上顶点
    points=np.array(PointList)  # 倾斜矩形
    outter_rect_l_t = np.append(np.min(points[::, 0]), np.min(points[::, 1]))
    # 求距离最大外接矩形左上点最近的点，作为给定四边形的左上顶点
    # 这一步应当是np.argmin(np.sqrt(np.sum(np.square(points - (x, y)), axis=1)))
    # 但是开不开算数平方根结果都一样，不是特别有必要，还浪费算力，就省了
    l_t_point_index = np.argmin(np.sum(np.square(points - outter_rect_l_t), axis=1))
    # 分别拿出来左上角点和剩余三点
    l_t_point = points[l_t_point_index]
    other_three_points = np.append(points[0:l_t_point_index:],
                                   points[l_t_point_index + 1::],
                                   axis=0)
    # 以x轴(此处以(1, 0)矢量视为x轴)为基准，根据剩余三点与x轴夹角角度排序
    BASE_VECTOR = np.asarray((1, 0))
    BASE_VECTOR_NORM = 1.0  # np.linalg.norm((1, 0))结果为1
    other_three_points = sorted(other_three_points,
                                key=lambda item: np.arccos(
                                    np.dot(BASE_VECTOR, item) /
                                    (BASE_VECTOR_NORM * np.linalg.norm(item))),
                                reverse=False)
    sorted_points = np.append(l_t_point.reshape(-1, 2),
                              np.asarray(other_three_points),
                              axis=0)
    return sorted_points

# Define a function that rotates a point around a center by an angle
def rotatePoint(xc, yc, xp, yp, theta):
    # Calculate the offset of the point from the center
    xoff = xp - xc
    yoff = yp - yc
    # Calculate the cosine and sine of the angle
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    # Apply the rotation matrix to the offset vector
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    # Add the rotated offset to the center coordinates and round to integers
    return (round(float(format(xc + pResx, '.0f')))), (round(float(format(yc + pResy, '.0f'))))

def voc_to_dota(xml_path, xml_name):
    txt_name = xml_name[:-4] + '.txt'
    txt_path = xml_path # + '/txt_label'
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    txt_file = os.path.join(txt_path, txt_name)
    file_path = os.path.join(xml_path, xml_name)
    tree = ET.parse(os.path.join(file_path))
    root = tree.getroot()

    with open(txt_file, 'w+', encoding='UTF-8') as out_file:

        # for obj in objects:
        #     if obj.find('type').text == 'robndbox':
        #         rbox_node = obj.find('robndbox')
        #         cat = obj.find('name').text
        #         rbox = dict()
        #         for key in ['cx', 'cy', 'w', 'h', 'angle']:
        #             rbox[key] = float(rbox_node.find(key).text)
        #         boxes.append(list((*rbox.values(), cat)))

        for obj in root.findall('object'):
            if obj.find('type').text == 'robndbox':
                name = obj.find('name').text
                # difficult = obj.find('difficult').text
                rbox_node = obj.find('robndbox')
                # robndbox = obj.find('robndbox').text
                cx = round(float(rbox_node.find('cx').text))
                cy = round(float(rbox_node.find('cy').text))
                w = round(float(rbox_node.find('w').text))
                h = round(float(rbox_node.find('h').text))
                angle = float(rbox_node.find('angle').text)

                p0x, p0y = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
                p1x, p1y = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
                p2x, p2y = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
                p3x, p3y = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)

                # 順序調整成 x1,y1,x2,y2,x3,y3,x4,y4
                # data = str(p1x) + ", " + str(p1y) + ", " + \
                #        str(p2x) + ", " + str(p2y) + ", " + str(p3x) + ", " + str(p3y) + ", " + str(p0x) + ", " + str(p0y)  + ", "
                data = [[p0x, p0y], [p1x, p1y], [p2x, p2y], [p3x, p3y]]
                data = sort_points_clockwise(data) # 逆时针排序
                p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y = [j for sub in data for j in sub]
                out_file.write('{},{},{},{},{},{},{},{},{}'.format(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, name + "\n"))

            elif obj.find('type').text == 'bndbox':  # for 4 point no angle format
                name = obj.find('name').text
                # difficult = obj.find('difficult').text
                box_node = obj.find('bndbox')
                xmin = round(float(box_node.find('xmin').text))
                ymin = round(float(box_node.find('ymin').text))
                xmax = round(float(box_node.find('xmax').text))
                ymax = round(float(box_node.find('ymax').text))
                data = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                data = sort_points_clockwise(data) # 逆时针排序
                p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y = [j for sub in data for j in sub]
                out_file.write('{},{},{},{},{},{},{},{},{}'.format(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, name + "\n"))


if __name__ == '__main__':
    # files_path = r'C:\develop\DBNet_pytorch_Wenmu_data\0_manual_prepare\short_and_less'
    # print(files_path)
    # # txt_file = 'test/presentation_032.txt'
    # roLabelImg2DOTA(files_path)
    # print('process {} to labels folder'.format(files_path))


    root_path = r'C:\develop\annotation_utils\procdataset'
    # root_path = r'C:\develop\annotation_utils\procdataset'
    # xml_file = 'TheWorldAhead2023pub20221209_114.xml'
    file_list = os.listdir(root_path)
    for i in file_list:
        if ('.xml' in i or '.XML' in i):
            print('process file {} to txt'.format(i))
            voc_to_dota(root_path, i)






    # test()
    # xml_path = '先探週刊第222期拜登訪台積電信賴產業鏈值千金2022-12-08i_001.xml'
    # img_path = '先探週刊第222期拜登訪台積電信賴產業鏈值千金2022-12-08i_001.jpg'
    # roLabelImg2DOTA(xml_path)
    # convert_rolabelimg2_4point(xml_path)
    # roLabelImg2DOTA('./')

'''
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
            # points = np.concatenate((points[:2], points[4:6], points[2:4], points[6:]),axis=0)  # [x1, y1, x3, y3, x2, y2, x4, y4] -> [x1, y1, x2, y2, x3, y3, x4, y4]
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
'''
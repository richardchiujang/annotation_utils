'''
使用順序
1. 將欲標註樣本用 DBNet inference ( DBNet.pytorch/input > DBNet.pytorch/output/inference )
2. 將 圖片樣本(jpg) + 剛剛預測的輸出(txt)，搬到 ./procdataset 下面並執行 dbnet_4_point_to_robndbox_cx_cy_w_h_angle_xml.py 
3. 同目錄下會產生 xml 檔案，打開 rolabelimg tool 將預測的標註修正成正確的
4. 執行 convert_rolabelimg_xml_to_4point.py ，會將 xml 轉成 txt 位置在 ./labels 裡面
5. 將 img_jpg + gt_txt (依自己分配)放到 C:\applications\datasets\(selfdata、selfdata_gt、selfdata_test、selfdata_gt_test)
6. 這樣可以持續增加訓練樣本

convert dbnet 4 point x1,y1,x2,y2,x3,y3,x4,y4,class txt file to rolabelimg xml format
help(cacudegree_xml())
help(euclideanDistance())
因為要填入園圖像大小 ww,hh,dd 如 (1000,1200,3)，所以需要讀取圖像(jpg)來產生xml的資料
作用為當 robndbox xml 要轉成 4 point dbnet annotation txt 要用原圖像尺寸計算位置，那時候就不用再次讀取影像

'''
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import SubElement
import xml.etree.ElementTree as ET

def polygonToRotRectangle_batch(bbox, with_module=False):
    '''
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
            shape [num_boxes, 8]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
            shape [num_rot_recs, 5]

    This function encloses the text box in a horizontal rectangle in text detection, including calculation of text box angle,
    calculation of rectangle center, matrix rotation and other operations. The specific operations of the function are as follows:
    First, the incoming bbox is reassembled into a 2x4 array, with the first row being 4 x-coordinates and the second row being 4 y-coordinates, for subsequent calculations.
    Determine whether the text box has an angle. If it does, calculate the angle value. If not, set the angle value to 0. The angle value is calculated using numpy's arctan2 function, which takes the slopes of the two sides of the text box into account and returns the angle in radians.
    Calculate the center of the rectangle, which is the average of the coordinates of the four corners.
    Create a rotation matrix, using the standard formula for rotation matrices, with the angle obtained in step 2.
    Perform matrix rotation to obtain the rotated text box coordinates.
    Calculate the coordinates of the four corners of the rectangle, which are the leftmost, rightmost, top, and bottom points. The specific method is to calculate the minimum and maximum values of the x and y coordinates respectively.
    Return the horizontal rectangle coordinates and angle information for the text box.            
    下面這段程式碼的作用是計算出文字框的角度。輸入的bbox是一個包含文字框四個角座標的3D陣列，其中第一維表示文字框的數量，
    第二維表示座標點的索引(從左上角開始順時針編號，左上角為0，右上角為1，右下角為2，左下角為3)，第三維表示x或y座標。
    程式碼首先判斷文字框是否有角度，若有則計算角度值，若無則將角度值設為0。計算角度值的方法是使用numpy的arctan2函數，將文字框的兩個邊的斜率帶入計算，並返回弧度值。
    另外，程式碼也包含一些特別的判斷，用來處理細長的文字框。因為這種文字框轉換成角度後可能會失真，所以程式碼會將角度較小的文字框的角度值設為0，
    以避免文字框顯示不正確。這些判斷條件是使用pi值和arctan2函數返回值進行計算的。
    ''' 

    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(-1, 2, 4),order='F')   # 分成 x1~x4 ,y1~y4
    # print(bbox)
    # print(bbox[:, 0,1], bbox[:, 0,0], bbox[:, 1,1], bbox[:, 1,0])
    # angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])  # 計算出角度
    if (bbox[:, 0,1]-bbox[:, 0,0])==0 or (bbox[:, 1,1]-bbox[:, 1,0])==0:    ### 判斷如果一邊有0，代表沒有角度，直接給angle=0值
        angle=np.array([-0.])
    else:
        angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])  # 計算出角度
        # 半圓分成16等分，範圍從 -0. ~ -pi，因為函式計算的角度都是負值
        if angle >= - np.pi*1/16 or - np.pi*7/16 >= angle >= -np.pi*9/16 or angle <= - np.pi*15/16:
            # print('angle: ', angle, type(angle))
            angle=np.array([-0.])
        else:
            pass  
    center = np.zeros((bbox.shape[0], 2, 1))
    for i in range(4):
        center[:, 0, 0] += bbox[:, 0,i]  # x 將四個點加起來 之後再除以四
        center[:, 1, 0] += bbox[:, 1,i]  # y 將四個點加起來 之後再除以四
    center = np.array(center,dtype=np.float32)/4.0

    '''
    這段程式碼是用來將方框的位置和角度進行旋轉的，其中的 R 是旋轉矩陣，透過這個矩陣的轉換，可以將原本的方框進行旋轉。
    這裡的角度是以弧度為單位，透過 np.cos 和 np.sin 函數計算正弦和餘弦值，生成旋轉矩陣 R。
    而 normalized 是透過矩陣運算將旋轉後的方框進行坐標轉換，使得原點移到方框的中心位置，即透過 center 讓方框的中心位於坐標系原點。
    最後，透過 np.matmul 函數計算旋轉矩陣 R 和中心化的方框 normalized 的乘積，得到最終的旋轉後方框的位置。

    這段程式主要是用來計算旋轉後的邊界框（bounding box）的最小外接矩形（minimum bounding rectangle），以便後續對旋轉後的圖像進行裁剪。具體來說：
    定義旋轉矩陣R。R矩陣是一個2x2的矩陣，其中元素值由角度angle決定。這個矩陣可以將平面上的點繞著原點旋轉angle角度。
    計算經過旋轉後的邊界框bbox的坐標，並標準化到中心點center。這裡用到了np.matmul函數來實現矩陣相乘。
    接下來，對經過旋轉和標準化的bbox在x、y軸上進行最大值和最小值的計算，從而得到最小外接矩形的四個頂點坐標。這些坐標值將用於對圖像進行裁剪。
    在計算時使用了np.min和np.max函數對坐標進行計算，而axis參數用於指定計算軸。np.min(normalized[:, 0, :], axis=1)的意思是對第1維的每個數據
    （即每個bbox）在第2維（即x軸）上取最小值，最後返回一個1維的結果。
    同理，np.max(normalized[:, 0, :], axis=1)的意思是對每個bbox在x軸上取最大值，最後返回一個1維的結果。
    而normalized[:, 0, :]的寫法則是取normalized矩陣的第2維（即x軸）上的所有數據，對於每個bbox取出來。
    最後再對這些數據進行最大值或最小值的計算。y軸上的計算則類似。    
    '''
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    normalized = np.matmul(R.transpose((2, 1, 0)),bbox-center)
 
    xmin = np.min(normalized[:, 0, :], axis=1)
    xmax = np.max(normalized[:, 0, :], axis=1)
    ymin = np.min(normalized[:, 1, :], axis=1)
    ymax = np.max(normalized[:, 1, :], axis=1)
 
    w = xmax - xmin + 3
    h = ymax - ymin + 3
    # # 因為辨識結果字框偏小，所以稍微加大一點
    # if w > h: # 比較寬扁
    #     w = w*1.05
    #     h = h*1.1
    # elif h > w: # 比較細高
    #     w = w*1.1
    #     h = h*1.05
    # else:
    #     pass # 暫時不管
    '''
    這段程式碼是將經過旋轉後的邊界框轉換為包含邊界框中心坐標、寬度、高度和旋轉角度的新形式。
    首先，w 和 h 分別是所有邊界框的寬度和高度，透過 [:, np.newaxis] 這個技巧將它們轉換為列向量，方便之後的矩陣拼接。
    接下來，如果參數 with_module 為 True，則旋轉角度 angle 會取模 $2\pi$，否則不變。這應該是為了限制旋轉角度的範圍，避免出現不必要的計算誤差。
    接著，dboxes 用 np.concatenate 函數拼接了多個數組，生成一個二維數組。
    其中，center[:, 0].astype(np.float64) 和 center[:, 1].astype(np.float64) 是中心點的橫、縱坐標，
    由於 np.zeros 創建的數組默認是浮點型，所以需要轉換為 np.float64 類型。w 和 h 分別是邊界框的寬度和高度，angle 是邊界框的旋轉角度，都是列向量。
    最後，np.concatenate 函數使用 axis=1 參數將多個列向量拼接為一個行向量，生成了一個形如 (N, 5) 的數組，
    其中 N 是邊界框的數量。每一行都對應一個邊界框，依次包含中心點的橫、縱坐標、寬度、高度和旋轉角度，即新的邊界框表示。最後，將 dboxes 返回即可。
    '''
    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    # TODO: check it
    if with_module:
        angle = angle[:, np.newaxis] % ( 2 * np.pi)
    else:
        angle = angle[:, np.newaxis]
    dboxes = np.concatenate((center[:, 0].astype(np.float64), center[:, 1].astype(np.float64), w, h, angle), axis=1)
    return dboxes     

def fourp_gen_xml(txt_file):
    """
    這是一個用於產生圖像的 XML 標註檔案的函式，該函式會根據 txt_file 中的坐標信息來計算四個點和其他相關信息並生成 XML 標註檔案。下面是這個函式的功能解釋：
    定義一個空的列表 rst，用於存儲 XML 標註檔案中的坐標信息；
    從 txt_file 中讀取標註信息，這裡的 txt_file 是一個 .txt 格式的標註文件；
    獲取影像文件的路徑和名稱，以讀取影像文件的大小，這裡假設圖像與標註在同一個路徑下，且圖像名稱和標註文件名稱相同，只有副檔名不同；
    讀取影像文件並獲取其大小；
    根據標註信息和圖像大小來計算四個點和其他相關信息，並將這些信息存儲在 rst 列表中；
    創建 XML 標註檔案，並將 rst 列表中的坐標信息添加到 XML 標註檔案中；
    將 XML 標註檔案寫入磁盤文件，文件名與 txt_file 相同，只是副檔名不同；
    打印信息，提示 XML 標註檔案已生成。
    """
    rst = []  # init annotation list
    filename = txt_file.split('\\')[-1].split('.txt')[0]  # anntation txt file name
    imgname = txt_file.split(filename)[0]+filename+'.jpg'  # image name for get img shape
    im = plt.imread(imgname)
#     plt.imshow(im)
    hh,ww,dd = im.shape
#     print(ww,hh)    
    with open(txt_file, encoding='utf-8') as file:
        for line in file:  
            line = [round(float(line),2) for line in line.replace('\n','').split(',')[:-1]]  # 最後一個可能是 字串 先所以不要 最後加回 words
            line.append('words')  # 最後加回 words
            rst.append(line) #storing everything in memory!
          
        a = ET.Element('annotation')
        b = ET.SubElement(a, 'folder').text = "unknow"
        b = ET.SubElement(a, 'filename').text = "unknow"
        b = ET.SubElement(a, 'path').text = 'unknow'
        b = ET.SubElement(a, 'source')
        c = ET.SubElement(b, 'database').text = 'unknow'
        b = ET.SubElement(a, 'size')
        c = ET.SubElement(b, 'width').text = '%d' % ww
        c = ET.SubElement(b, 'height').text = '%d' % hh
        c = ET.SubElement(b, 'depth').text = '%d' % dd
        b = ET.SubElement(a, 'segmented').text = '0'    

        # =====================================
        for x1,y1,x2,y2,x3,y3,x4,y4,_ in rst: 
            # cx, cy = float((max(x1,x2,x3,x4)+min(x1,x2,x3,x4))/2), float((max(y1,y2,y3,y4)+min(y1,y2,y3,y4))/2)  # 計算中心點
            # w, h = float(euclideanDistance(x4,y4,x3,y3)), float(euclideanDistance(x2,y2,x3,y3))   # (x1,y1,x2,y2) 改成 (x4,y4,x3,y3)
            # angle = cacudegree_xml(x4,y4,x3,y3)
            bbox = x1,y1,x2,y2,x3,y3,x4,y4
            dboxes = polygonToRotRectangle_batch(bbox)
            cx, cy, w, h, angle = dboxes[0]
            b = ET.SubElement(a, 'object')
            c = ET.SubElement(b, 'type').text = 'robndbox'
            c = ET.SubElement(b, 'name').text = 'words'
            c = ET.SubElement(b, 'pose').text = 'Unspecified'
            c = ET.SubElement(b, 'truncated').text = '0'
            c = ET.SubElement(b, 'difficult').text = '0'
            c = ET.SubElement(b, 'robndbox')
            d = ET.SubElement(c, 'cx').text = '%d' % cx
            d = ET.SubElement(c, 'cy').text = '%d' % cy
            d = ET.SubElement(c, 'w').text = '%d' % w
            d = ET.SubElement(c, 'h').text = '%d' % h
            d = ET.SubElement(c, 'angle').text = '%f' % angle

        ### =============================================
        ET.indent(a, space='  ', level=0)
        tree = ElementTree(a)
        # ET.dump(a)

        with open(txt_file.split(filename)[0]+'{}.xml'.format(filename), 'wb') as f:
            tree.write(f, encoding='UTF-8', xml_declaration=True) 
        print('generate {}{}.xml file.'.format(txt_file.split(filename)[0], filename))   


if __name__ == '__main__':
    """
    使用順序
    1. 將欲標註樣本用 DBNet inference ( DBNet.pytorch/input > DBNet.pytorch/output/inference )
    2. 將 圖片樣本(jpg) + 剛剛預測的輸出(txt)，搬到 ./procdataset 下面並執行 dbnet_4_point_to_robndbox_cx_cy_w_h_angle_xml.py 
    3. 同目錄下會產生 xml 檔案，打開 rolabelimg tool 將預測的標註修正成正確的
    4. 執行 convert_rolabelimg_to_4point.py ，會將 xml 轉成 txt 位置在 ./labels 裡面
    5. 將 img_jpg + gt_txt (依自己分配)放到 C:\applications\datasets\(selfdata、selfdata_gt、selfdata_test、selfdata_gt_test)
    6. 這樣可以持續增加訓練樣本
    
    用 DBNet.pytorch inferencr ( input/*.jpg > output/inference/*.txt) 結果都COPY到 C:\applications\annotation_tools\annotation_utils\procdataset
    需要影像檔 jpg 因為要讀取 image size for xml 
    會將 file_path 內的 txt 轉成 xml 在同一個路徑下
    可以用 roLabelImg 工具檢查一下，同時調整(修正)錯誤的label
    再用 convert_rolabelimg_to_4point.py 產生新的 ./labels/filename.txt for DBNet ground truth txt files.
    再將 image & gt txt 分配到 dataset selfdata 的 train test 裡面 
    """
    files_path = glob.glob('procdataset/*.txt')
    print(files_path)
    # txt_file = 'test/presentation_032.txt'
    for txt_file in files_path:
        fourp_gen_xml(txt_file) 



# def cacudegree_xml(x4,y4,x3,y3):
#     """
#     這段程式的作用是計算文字框的傾斜角度，輸入的是文字框的前兩個點和後兩個點的坐標 (x3,y3) 和 (x4,y4)，
#     程式會根據這兩個點的坐標計算出文字框的傾斜角度，並以弧度的形式返回該角度值。
#     使用了 Python 內建的 math 函式庫中的 atan2 函數，用來計算反正切值，求出由 (x4,y4) 到 (x3,y3) 連線和 x 軸正方向之間的夾角，
#     這個夾角就是文字框的傾斜角度。最後將計算出來的角度以弧度的形式返回。
#     """
#     angle=math.atan2(y3-y4,x3-x4)
# #     degree=math.degrees(degree)
# #     degree=math.degrees()    
#     return angle

# def euclideanDistance(x1,y1,x2,y2):
#     """
#     這段程式是一個計算兩點之間歐幾里德距離的函式。歐幾里德距離是指在平面上，兩個點之間的距離，也稱作直線距離。
#     這個函式輸入四個變數，分別是兩個點的 x 和 y 座標，然後透過歐幾里德距離公式 $\sqrt{(x1-x2)^2+(y1-y2)^2}$，
#     計算出兩點之間的距離，最後回傳這個距離。
#     """
#     return math.sqrt(((x1-x2)**2)+((y1-y2)**2) )       


# if (bbox[:, 0,1]-bbox[:, 0,0])==0 or (bbox[:, 1,1]-bbox[:, 1,0])==0:    ### 判斷如果一邊有0，代表沒有角度，直接給angle=0值
#     angle=np.array([-0.])
# else:
#     angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])  # 計算出角度
#     # 半圓分成16等分，範圍從 -0. ~ -pi，因為函式計算的角度都是負值
#     if angle >= - np.pi*1/16 or - np.pi*7/16 >= angle >= -np.pi*9/16 or angle <= - np.pi*15/16:
#         # print('angle: ', angle, type(angle))
#         angle=np.array([-0.])
#     else:
#         pass    
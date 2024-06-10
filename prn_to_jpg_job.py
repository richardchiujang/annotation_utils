#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   prn_to_jpg_job.py
@Time    :   2022/12/13 15:53
@Author  :   richard
@Version :   1.0
@Contact :   rchiu3210@gmail.com
@License :   MIT LICENSE
@Desc    :   <awaiting description>
2024-05-25 整個邏輯重新寫 
1.產生包含分類及文件名的資料夾
2.每分文件亂數刪除50張以外的數量
3.移掉exif資訊，然後逐資料夾人工翻轉方向
'''
# 複製更新到 C:\develop\annotation_utils 中 for github commit

import glob, time, os, random
import win32print, win32api
import pyperclip
import logging
import tqdm

pHandle = win32print.GetDefaultPrinter() 
print(pHandle)

logger = logging.getLogger(__name__)
logging.basicConfig(filename='runlog.txt', encoding='utf-8', level=logging.DEBUG)

print('start processing prn to jpg')

### 1.先把prn檔轉成jpg
def prn_to_jpg(dict_prn):   
    for key, value in tqdm.tqdm(dict_prn.items()):
        logger.debug('key, value[0], value[1]: %s %s %s', key, value[0], value[1])
        # DEBUG:__main__:key, value[0], value[1]: 遠見雜誌 202308[第446期] 重新設計你的百歲人生.prn .\prn_doc_all 雜誌_繁中
        fname = key.split('.prn')[0]
        replace_list = ['[', ']', ' ', '(', ')', '【', '】', '.', ',', '《', '》', '、', '。', '「', '」', '『', '』']
        for x in replace_list:
            fname = fname.replace(x, '')
        logger.debug('fname: %s', fname)
        fdname = value[1]
        rootdir = value[0]
        # '.\output\商業市場金融分析\2023中小及新創企業財務資源應用手冊\2023中小及新創企業財務資源應用手冊'
        target_path = r'.\output\{}\{}\{}'.format(fdname, fname, fname) 
        # '.\prn_doc_all\雜誌_繁中\遠見雜誌 202308[第446期] 重新設計你的百歲人生.prn'
        source_path = r'{}\{}\{}'.format(rootdir, fdname, key)
        logger.debug('fname, fdname, rootdir, target_path, source_path: %s %s %s %s %s', fname, fdname, rootdir, target_path, source_path)

        os.makedirs(r'.\output\{}'.format(fdname), exist_ok=True)
        os.makedirs(r'.\output\{}\{}'.format(fdname, fname), exist_ok=True)
        dpis = ['300']   
        for dpi in dpis:
            param = '-sDEVICE=jpeg -r{} -dDownScaleFactor=3 -dJPEGQ=100 -dQFactor=1 -o "{}_{}_%03d.jpg" "{}"'.format(dpi, target_path, dpi, source_path)
            logger.info('param: %s', param)
            os.system("gs.exe {}".format(param))

# glob  full path and filename
# prn_files = glob.glob(r'.\prn_doc_all\*\*.prn')
def dict_prn():
    dict_prn = {}
    for dirpath, dnames, fnames in os.walk(r'.\prn_doc_all'):
        # logger.debug('now record: %s %s %s', dirpath, dnames, fnames)
        if fnames==[]:
            rootdir = dirpath
            fdlist = dnames
            # logger.debug('rootdir, folder list: %s %s', rootdir, fdlist)    
            fdcont = 0
        if dnames==[]:
            for fname in fnames:
                # logger.debug('filename, foldername, rootdir: %s %s %s', fname, fdlist[fdcont], rootdir)
                dict_prn[fname] = [rootdir, fdlist[fdcont]]
            fdcont += 1
    return dict_prn

# logger.debug('dict_prn: %s', dict_prn)


logger.info('start processing prn to jpg')
prn_to_jpg(dict_prn())
logger.info('done\n')



### 2. 依照資料夾內容亂數刪除過多的jpg檔，讓資料避免偏斜
import numpy as np
def del_jpg(dir):
    for root, dirs, files in os.walk(dir):
        if not files:
            # print(root, len(dirs))
            for dir in dirs:
                glob_files = glob.glob(os.path.join(root, dir, '*.jpg'))
                # print(len(glob_files), dir)
                if len(glob_files) > 50:
                    # print(len(glob_files), glob_files[-2:])
                    file_sum = len(glob_files)
                    del_sum = file_sum - 50 # random delete over 50 files
                    item_list = np.arange(file_sum)
                    random.shuffle(item_list)
                    # print(item_list)
                    cnt = 0
                    for x in range(del_sum): # delete del_sum number of files
                        os.remove(glob_files[item_list[x]])
                        cnt += 1
                    print('deleted:', cnt)

del_jpg('output')

# ### 這裡怎轉都沒用所以跳過不做
# # 3.把jpg檔的exif資訊清除
# import tqdm
# # from PIL import Image
# import piexif, exifread
# 
# class Info:
#     ImageWidth = ""
#     ImageLength = ""
#     Make = ""
#     Model = ""
#     GPSLatitudeRef = ""
#     GPSLatitude = ""
#     GPSLongitudeRef = ""
#     GPSLongitude = ""
#     DateTimeOriginal = ""
#     Orientation = ""
#     # def to_string(self):
#         # print('\n'.join(['%s:%s' % item for item in self.__dict__.items()]))
# 
# def parse_image(path):
#     """解析单张图片的信息"""
#     file = open(path, "rb")
#     tags = exifread.process_file(file)
#     info = Info()
#     for tag in tags.keys():
#         if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
#             print("Key: %s, value %s" % (tag, tags[tag]))
#             info.ImageWidth = tags["Image ImageWidth"]
#             info.ImageLength = tags["Image ImageLength"]
#             info.Make = tags["Image Make"]
#             info.Model = tags["Image Model"]
#             info.GPSLatitudeRef = tags["GPS GPSLatitudeRef"]
#             info.GPSLatitude = tags["GPS GPSLatitude"]
#             info.GPSLongitudeRef = tags["GPS GPSLongitudeRef"]
#             info.GPSLongitude = tags["GPS GPSLongitude"]
#             info.DateTimeOriginal = tags["EXIF DateTimeOriginal"]
#             info.Orientation = tags["Image Orientation"]
#     return info
# 
# def earse_exif(dir):
#     for root, dirs, files in os.walk('output'):
#         for name in tqdm.tqdm(files):
#             file = os.path.join(root, name)
#             if file.endswith("jpg") or file.endswith("jpeg"):
#                 piexif.remove(file)
#                 # parse_image(file).to_string()
# 
# earse_exif('output')
# print('done')
# 
# # parse_image(r'output\0_另外準備\short_and_less\OCR_SourceSize12_2_landscape_細明體300.jpg')

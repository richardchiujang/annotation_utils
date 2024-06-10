# ENCODING="UTF-8"
"""
從指定路徑下將資料(更名prefix)搬到目的資料夾，並產生清單
TD_TR、ch4_test_images(ICDAR2015)
./prepare/train
               /img
               /gt
./prepare/test
              /img
              /gt
./prepare/train.txt
./prepare/test.txt
"""

import pathlib
import glob, os, shutil
import random

def gendataset(base_path, images_path, gt_path, prefix, mode, num=None):
    """
    mode = {train; test}
    prefix like {ICDAR2015; TD500; }
    修改新增 shuffle 數量改成1個取樣數字(量)，不是原來的區間
    
    """
    # basename = pathlib.Path('C:/applications/datasets')
    k = 0
    out_file = '{}/prepare/{}_{}.txt'.format(base_path, prefix, mode)  # base_path/prepare/ICDAR2015_train.txt
    out_list = [] 
    imgs_path = str(pathlib.Path(base_path).joinpath(images_path)) + '/*.*'
    imgs = [os.path.basename(x) for x in glob.glob(imgs_path)]
    random.shuffle(imgs)
    imgs = imgs[:num] # 取樣數量

    ### 將 gts 取樣篩選成跟 imgs 一致 ### 直接用 image name 產生 txt name
    ### 根據資料集不同 對應名稱有調整
    if prefix == 'ICDAR2015':
        gts = ['gt_' + x[:-4] + '.txt' for x in imgs]
    elif prefix == 'TD500' or prefix == 'TR400':
        gts = [x[:-4]+'.JPG.txt' for x in imgs]  
    else:
        gts = [x[:-4]+'.txt' for x in imgs]  

    for i, j in zip(imgs, gts):
        if (i.split('.')[-1]).upper() != 'JPG':
            pass   
        else:
            src = pathlib.Path(base_path).joinpath(images_path, i)
            dst = str(pathlib.Path(base_path).joinpath('prepare', mode, 'img'))+'\{}_{}'.format(prefix, i)
#             print(dst)
            shutil.copyfile(src, dst)
            src = pathlib.Path(base_path).joinpath(gt_path, j) 
            dst = str(pathlib.Path(base_path).joinpath('prepare', mode, 'gt')) + '\{}_{}'.format(prefix, j)
            shutil.copyfile(src, dst)   
            # generate file list
            line = './datasets/{}/img/{}_{}\t./datasets/{}/gt/{}_{}'.format(mode, prefix, imgs[k], mode, prefix, gts[k])
            out_list.append(line)
        k+=1
    print('number of {} {} images : {}'.format(prefix, mode, k))
            
    with open(out_file, 'w', encoding="UTF-8") as fp:
        for item in out_list:
            fp.write("%s\n" % item)

def gen_list_txt(base_path, mode, file_lists):
    """
    mode {test; train}
    file_lists = [file1, file2, ...] 
    """
    gen_file = base_path + '/prepare/{}.txt'.format(mode)
    filenames = file_lists
    with open(gen_file, 'w', encoding="UTF-8") as outfile:
        for fname in filenames:
            fname_path = base_path + '/prepare/{}'.format(fname)
            with open(fname_path, encoding="UTF-8") as infile:
                outfile.write(infile.read())
            os.remove(fname_path)

# 在 datasets/prepare 下建立路徑架構
def mk_folder(base_path):
    # base_path = pathlib.Path('C:/applications/datasets')
    path_list = ['prepare/train', 'prepare/train/img', 'prepare/train/gt', 'prepare/test', 'prepare/test/img', 'prepare/test/gt']
    # mkdir [prepare] folder if not exsit.
    if not pathlib.Path.is_dir(pathlib.Path(base_path).joinpath('prepare')):
        print('mkdir ', pathlib.Path(base_path).joinpath('prepare'))
        pathlib.Path.mkdir(pathlib.Path(base_path).joinpath('prepare')) 
    # mkdir others if not exsit.
    for p in path_list:
        if not pathlib.Path.is_dir(pathlib.Path(base_path).joinpath(p)):
            print('mkdir ', pathlib.Path(base_path).joinpath(p))
            pathlib.Path.mkdir(pathlib.Path(base_path).joinpath(p))

# # 'selfdata', 'selfdata_gt', 'selfdata'
def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--base_path', default=r'C:/Application/Develop/datasets', type=str)
#     parser.add_argument('--img_folder', default='selfdata', type=str, help='source img folder')
#     parser.add_argument('--gt_folder', default='selfdata_gt', type=str, help='source gt folder')
#     parser.add_argument('--thre', default=0.3,type=float, help='the thresh of post_processing')
#     parser.add_argument('--polygon', action='store_true', help='output polygon or box')
#     parser.add_argument('--show', action='store_true', help='show result')
#     parser.add_argument('--save_resut', action='store_true', help='save box and score to txt file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    這會將 C:/Application/Develop/datasets 裡面指定的資料集，處理並搬到 C:/applications/Develop/datasets/prepare 中
    分別放在 /train/img、/train/gt、/test/img、/test/gt
    並產生 train.txt test.txt 兩分清單文件 
    """
    import pathlib
    from tqdm import tqdm
    # import matplotlib.pyplot as plt

    args = init_args()
    print(args)
    print(pathlib.Path.cwd())
    mk_folder(args.base_path)

    ### ICDAR2015
    # gendataset(args.base_path, 'ch4_test_images', 'Challenge4_Test_Task1_GT', 'ICDAR2015', 'test', num=50)
    # gendataset(args.base_path, 'ch4_training_images', 'ch4_training_localization_transcription_gt', 'ICDAR2015', 'train', num=100)
    # gendataset(args.base_path, 'TD_TR/TD_TR/TD500/train_images', 'TD_TR/TD_TR/TD500/train_gts', 'TD500', 'train', num=100)
    # gendataset(args.base_path, 'TD_TR/TD_TR/TD500/test_images', 'TD_TR/TD_TR/TD500/test_gts', 'TD500', 'test', num=50)
    # gendataset(args.base_path, 'TD_TR/TD_TR/TR400/train_images', 'TD_TR/TD_TR/TR400/train_gts', 'TR400', 'train', num=100)

    # gendataset(args.base_path, 'ImagesPart1/ImagesPart1', 'train_gt_t13', 'ICDAR2019', 'train', num=100)
    # gendataset(args.base_path, 'ImagesPart1_test', 'train_gt_t13_test', 'ICDAR2019', 'test', num=50)
    gendataset(args.base_path, 'selfdata', 'selfdata_gt', 'selfdata', 'train', num=None)
    gendataset(args.base_path, 'selfdata_test', 'selfdata_test_gt', 'selfdata', 'test', num=None)
    # gendataset(args.base_path, 'selfdata_vertical', 'selfdata_vertical_gt', 'selfdata_vertical', 'train', num=None)        ### 100
    # gendataset(args.base_path, 'selfdata_vertical_test', 'selfdata_vertical_test_gt', 'selfdata_vertical', 'test', num=None)    ### 17    
    print('gen train.txt , test.txt.')
    # gen_list_txt(args.base_path, 'test', ['selfdata_test.txt','selfdata_vertical_test.txt','ICDAR2019_test.txt'])
    # gen_list_txt(args.base_path, 'train', ['selfdata_train.txt','selfdata_vertical_train.txt','ICDAR2019_train.txt'])

    # gen_list_txt(args.base_path, 'test', ['selfdata_test.txt','selfdata_vertical_test.txt','ICDAR2019_test.txt','ICDAR2015_test.txt','TD500_test.txt'])
    # gen_list_txt(args.base_path, 'train', ['selfdata_train.txt','selfdata_vertical_train.txt','ICDAR2019_train.txt','ICDAR2015_train.txt','TD500_train.txt','TR400_train.txt'])

    # gen_list_txt(args.base_path, 'test', ['selfdata_test.txt','selfdata_vertical_test.txt'])
    # gen_list_txt(args.base_path, 'train', ['selfdata_train.txt','selfdata_vertical_train.txt'])

    gen_list_txt(args.base_path, 'test', ['selfdata_test.txt',])
    gen_list_txt(args.base_path, 'train', ['selfdata_train.txt',])

    # gen_list_txt(args.base_path, 'test', ['selfdata_test.txt','ICDAR2015_test.txt'])
    # gen_list_txt(args.base_path, 'train', ['selfdata_train.txt','ICDAR2015_train.txt'])

    # gen_list_txt(args.base_path, 'test', ['selfdata_test.txt','ICDAR2015_test.txt','TD500_test.txt','ICDAR2019_test.txt'])
    # gen_list_txt(args.base_path, 'train', ['selfdata_train.txt','ICDAR2015_train.txt','ICDAR2019_train.txt','TD500_train.txt','TR400_train.txt'])
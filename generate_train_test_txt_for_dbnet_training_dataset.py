import glob
import os, shutil
import random

jpgs = glob.glob(r'.\preprocess_dataset\datasets_mix\source\*.jpg')
print(len(jpgs))
# print(jpgs[:3])
random.shuffle(jpgs)

train_jpgs = jpgs[:int(len(jpgs)*0.6)]
test_jpgs = jpgs[int(len(jpgs)*0.6):]

# '.\\preprocess_dataset\\datasets_mix\\source\\01_300_001.jpg' 
with open(r'.\preprocess_dataset\datasets_mix\datasets\train.txt', 'w', encoding='utf-8') as f:
    for j in train_jpgs:
        jpg = j.replace('.\\preprocess_dataset\\datasets_mix\\source\\', '.\\datasets\\train\\img\\') # 這是文件內的路徑
        txt = jpg.replace('.jpg', '.txt').replace('img', 'gt') # 這是文件內的路徑
        xml = jpg.replace('.jpg', '.xml').replace('img', 'gt')
        f.write(jpg + '\t' + txt + '\n')
        shutil.move(j, '.\\preprocess_dataset\\datasets_mix\\' + jpg)
        shutil.move(j.replace('.jpg', '.txt'), '.\\preprocess_dataset\\datasets_mix\\' + txt)
        try:
            shutil.move(j.replace('.jpg', '.xml'), '.\\preprocess_dataset\\datasets_mix\\' + xml)
        except:
            pass
    f.close()

with open(r'.\preprocess_dataset\datasets_mix\datasets\test.txt', 'w', encoding='utf-8') as f:
    for j in test_jpgs:
        jpg = j.replace('.\\preprocess_dataset\\datasets_mix\\source\\', '.\\datasets\\test\\img\\') # 這是文件內的路徑
        txt = jpg.replace('.jpg', '.txt').replace('img', 'gt') # 這是文件內的路徑
        xml = jpg.replace('.jpg', '.xml').replace('img', 'gt')
        f.write(jpg + '\t' + txt + '\n')
        shutil.move(j, '.\\preprocess_dataset\\datasets_mix\\' + jpg)
        shutil.move(j.replace('.jpg', '.txt'), '.\\preprocess_dataset\\datasets_mix\\' + txt)
        try:
            shutil.move(j.replace('.jpg', '.xml'), '.\\preprocess_dataset\\datasets_mix\\' + xml)
        except:
            pass
    f.close()








# jpgs = glob.glob(r'.\preprocess_dataset\datasets\train\img\*.jpg')
# txts = glob.glob(r'.\preprocess_dataset\datasets\train\gt\*.txt')
# i = 0
# with open(r'.\preprocess_dataset\datasets\train.txt', 'w', encoding='utf-8') as f:
#     for jpg, txt in zip(jpgs, txts):
#         if os.path.basename(jpg).split('.jpg')[0] == os.path.basename(txt).split('.txt')[0]:
#             jpg = jpg.replace('preprocess_dataset\\', '')
#             txt = txt.replace('preprocess_dataset\\', '')
#             f.write(jpg + '\t' + txt + '\n')
#             i += 1
#         else:
#             print('error')
# f.close()
# print(i)

# jpgs = glob.glob(r'.\preprocess_dataset\datasets\test\img\*.jpg')
# txts = glob.glob(r'.\preprocess_dataset\datasets\test\gt\*.txt')
# i = 0
# with open(r'.\preprocess_dataset\datasets\test.txt', 'w', encoding='utf-8') as f:
#     for jpg, txt in zip(jpgs, txts):
#         if os.path.basename(jpg).split('.jpg')[0] == os.path.basename(txt).split('.txt')[0]:
#             jpg = jpg.replace('preprocess_dataset\\', '')
#             txt = txt.replace('preprocess_dataset\\', '')
#             f.write(jpg + '\t' + txt + '\n')
#             i += 1
#         else:
#             print('error')
# f.close()
# print(i)





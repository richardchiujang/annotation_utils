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
'''

import glob, time, os, numpy, random

def prn_to_jpg(files):
    for f in files:
        foname = (os.path.basename(f).split('.prn')[0]).replace(' ','')
        try:
            os.mkdir('./output/'+foname)
        except:
            pass
        # r=numpy.random.choice([200,300,400,500,600],1,p=[0.1, 0.3, 0.2, 0.2, 0.2])[0]
        r = 600   # r = 600, A4 size = 1653 x 2338 ; r = 300 , A4 size = 1169 x 826 , 
        # param = '-sDEVICE=jpeg -r{} -dDownScaleFactor=3 -dJPEGQ=100 -dQFactor=1 -o ".\\output\\{}\\{}_%03d.jpg" "{}"'.format(r, foname, foname, f)
        param = '-sDEVICE=jpeg -r{} -dDownScaleFactor=3 -dJPEGQ=100 -dQFactor=1 -o ".\\output\\all\\{}_%03d.jpg" "{}"'.format(r, foname, f)
        print(param,'\n')
        os.system("gs.exe {}".format(param))

files = glob.glob(r'.\prn_doc\*.prn')

random.shuffle(files)
print('process files num ', len(files),'\n')

prn_to_jpg(files)
    

## batch_rename_files.py
import os

def match_files(file_list, match_words, replace_words):
    count = 0
    for f in file_list:
        if f.find(match_words) !=-1:
            if f[-4:]=='.jpg':
                f_name = f.split('.jpg')[0]
                f_new_name = f_name[-3:]
                # print(f_new_name)
                os.rename('procdataset/'+f, 'procdataset/{}_{}.jpg'.format(replace_words, f_new_name))
            elif f[-4:]=='.xml':
                f_name = f.split('.xml')[0]
                f_new_name = f_name[-3:]
                # print(f_new_name)
                os.rename('procdataset/'+f, 'procdataset/{}_{}.xml'.format(replace_words, f_new_name))  
            else:
                f_name = f.split('.txt')[0]
                f_new_name = f_name[-3:]
                # print(f_new_name)
                os.rename('procdataset/'+f, 'procdataset/{}_{}.txt'.format(replace_words, f_new_name))
            count +=1
    return print(count)


if __name__=='__main__':

    file_list = os.listdir('procdataset')
    print(file_list[:10], len(file_list))
    match_files(file_list, 'Business.Today_1355-EV_', 'BusinessToday_1355-EV')
    match_files(file_list, 'Wealth財訊.674-台積電再造日本矽島2022-12-08_', 'Wealth_new_674-2022-12-08')
    match_files(file_list, '先探週刊第222期拜登訪台積電信賴產業鏈值千金2022-12-08i_', 'pioneeo_222_2022')
    match_files(file_list, 'Business.Today_1355-EV_', 'BusinessToday_135599-EV')
#-*- coding: UTF-8 -*-
import os
import  random
import shutil
import PIL.Image as Image

import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--ind' , default='./IMG_SRC_AUG/') 
parser.add_argument('--out' , default='./IMG_SRC_AUG_SHK/')
parser.add_argument('--ratio' , default='0.1')
parser.add_argument('--ran_seed' , default='5')

MINPIC = 20  #minimum number of shrunk picture

def mkdir(path):     #判断是否存在指定文件夹，不存在则创建
    # 引入模块
    import os
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
 
        print( path)
        print( ' 創建成功' )
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print( path)
        print( ' 目錄已存在' )
        return False
    
# 遍历指定目录，显示目录下的所有文件名

def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    child_file_name = []
    full_child_file_list = []
    for allDir in pathDir:
        #allDir =unicode(allDir, encoding = 'utf-8')  #python3 not need
        child = os.path.join('%s%s' % (filepath, allDir))
        #print child.decode('gbk') # .decode('gbk')是解决中文显示乱码问题
        full_child_file_list.append(child)
        child_file_name.append(allDir)
    return  full_child_file_list,child_file_name
 
def eachFile1(filepath):
    dir_list  = []
    name_list = []
    pathDir   =  os.listdir(filepath)
    for allDir in pathDir:
        name_list.append(allDir)
        child = os.path.join('%s%s' % (filepath+'/', allDir))
        dir_list.append(child)
    return  dir_list,name_list
 

    
def main(args): 
    
    args.ind = os.path.expanduser(args.ind)
    args.out = os.path.expanduser(args.out)
    
    SRCIMGD = args.ind+'/'
    IMGD    = args.out+'/'
    
    RATIO = args.ratio
    RANSD = args.ran_seed

    filePath,IMG_SRC_list = eachFile(SRCIMGD)
    
    for i in IMG_SRC_list:
        path  = './IMG_9/' +i
        mkdir(path)
        
        path = IMGD + i
        mkdir(path)

    for i in filePath:
        pic_dir,pic_name = eachFile1(i)
        random.seed(RANSD)
        random.shuffle(pic_dir)
        
        NP = int( float(RATIO)*len(pic_dir) )  #num of picture
        if NP < MINPIC : 
            NP = MINPIC
            if NP > len(pic_dir):
                NP = len(pic_dir)
        small_list = pic_dir[0:NP]
        big_list   = pic_dir[NP: ]
        
        for j in big_list:   
            fromImage = Image.open(j)
            j = j.replace('IMG_SRC_AUG','IMG_9')
            fromImage.save(j)
            
        for k in small_list:
            fromImage = Image.open(k)
            k = k.replace('IMG_SRC_AUG','IMG_SRC_AUG_SHK')
            fromImage.save(k)


if __name__ == '__main__':            
    args = parser.parse_args()
    main(args)
    

   

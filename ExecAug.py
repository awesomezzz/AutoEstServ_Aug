from keras.applications.xception import Xception as xcmodel, preprocess_input as xcprep
from keras.applications.mobilenetv2 import MobileNetV2 as mv2model, preprocess_input as mv2prep
from keras.applications.nasnet import NASNetMobile as nasmodel, preprocess_input as nasmprep
from keras.applications.resnet50 import ResNet50 as resmodel, preprocess_input as resprep
from keras.applications.densenet import DenseNet169 as dsmodel, preprocess_input as dsprep

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import math, os, time, datetime
import argparse
import itertools
import pandas as pd

from itertools import combinations


current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()

# defining corresponding input/output images directories
parser.add_argument('--ind', default='IMG_SRC') 
parser.add_argument('--sh' , default='SugModels_SHK.sh')
parser.add_argument('--log' , default='SugModels_SHK.log')
parser.add_argument('--csv' , default='PreANA.csv')
parser.add_argument('--classes', '-c', default='./classes.txt') 
parser.add_argument('--result_root', '-r',  default='./result/')

#parser.add_argument('--aug' , default='(1, 2, 3, 4)')
#parser.add_argument('--augsize' , default='50')
parser.add_argument('--outd'    , default='IMG_SRC_AUG')
#parser.add_argument('--augnum'    , default='0')

# setting learning models parameters
parser.add_argument('--insp' , default='PreParams.py') 
parser.add_argument('--lr_pre' , type=float, default=1e-3)
parser.add_argument('--lr_fine', type=float, default=1e-4)

parser.add_argument('--split', type=float, default=0.8)


datagen = ImageDataGenerator(
        rotation_range=120 ,
        width_shift_range=0.2 ,
        height_shift_range=0.2 ,
        channel_shift_range=50 ,
        shear_range=0.2 ,
        zoom_range=0.2 ,
        horizontal_flip=True ,
        vertical_flip=True ,
        brightness_range=(0.7,0.7) ,
        rescale=1. / 255 ) 

def main(args):

    ### execute suggest parameter
    sugpfile = './'+ args.insp              # Suggested parameters file from command ...  
    exec(open(sugpfile).read(), globals())  # directely 'sourcing' the outside python prog.
    


    select_ct = args.classes
    
    with open(select_ct, 'r') as ct:
        classes = ct.readlines()
        classes = list(map(lambda x: x.strip(), classes))
    
    import os
    
    if os.path.exists(args.outd) == False:
        os.makedirs(args.outd)

  
    for classname in classes :
        inCWD  = args.ind  +'/'
      #  outCWD = args.outd + '_' + args.augnum +'/'
        outCWD = args.outd +'/'
        cnm    = classname
        
        ImgDir   = inCWD  + cnm  # Source Images Direcotry ...
        OutDir   = outCWD + cnm  # Output Directory
        if os.path.exists(OutDir) == False:
            os.makedirs(OutDir)
            
        SVPrefix = 'aug_'       # prefix for the out image files in output directory
        
      #  OutputSize = int(args.augsize)    # Here is the output size control!

      #  OutputSize = len(ImgDir)

      #  print("OutputSize : ", OutputSize)
        
        Flist = os.listdir( ImgDir )
        print("Flist : ", Flist)
        fs = len(Flist)  

        print("fs : ", fs)
        
        for i in range(fs) :
      #  for i in range(OutputSize) :
            j = i%fs
            img_file = ImgDir + '/' + Flist[j] # converting img file with id(i)
            img = load_img( img_file )    # this is a PIL image, please replace to your own file path
            x = img_to_array(img)          # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
            print( img_file )
            # the .flow() command below generates randomly transformed images and saves the results to [OutDir] directory
            for _ in datagen.flow( x, batch_size=1, save_to_dir = OutDir,  # strange way to generate a random images ... 
                     save_prefix=SVPrefix+i.__str__() , save_format='jpg'): break




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


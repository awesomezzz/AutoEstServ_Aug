from keras.applications.xception import Xception as xcmodel, preprocess_input as xcprep
from keras.applications.mobilenetv2 import MobileNetV2 as mv2model, preprocess_input as mv2prep
from keras.applications.nasnet import NASNetMobile as nasmodel, preprocess_input as nasmprep
from keras.applications.resnet50 import ResNet50 as resmodel, preprocess_input as resprep
from keras.applications.densenet import DenseNet169 as dsmodel, preprocess_input as dsprep

from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing import image
from keras.losses import categorical_crossentropy
from keras.layers import Dense, GlobalAveragePooling2D, Activation
from keras.models import Model, Sequential
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import math, os, time, datetime
import numpy as np

import argparse
import imghdr
import pickle as pkl

import itertools

import gc

from keras import backend as K

import tensorflow as tf 
import keras.backend.tensorflow_backend as KTF 

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn import metrics
from itertools import cycle
from scipy import interp
import pandas as pd


current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()

# defining corresponding input/output images directories
parser.add_argument('--ind', default='IMG_SRC') 
parser.add_argument('--inshlog', default='SugANA_SRC.log') 
parser.add_argument('--incsvone', default='SHKANA.csv') 
parser.add_argument('--incsvtwo', default='SugANA.csv') 
parser.add_argument('--sh' , default='SugANAs_FIN.sh') 
parser.add_argument('--log' , default='SugANA_FIN.log')
parser.add_argument('--csv' , default='FINANA.csv')
parser.add_argument('--result_root', '-r',  default='./result/')

# setting learning models parameters
parser.add_argument('--num_model', default= 5 ) 
parser.add_argument('--insp' , default='FinParams.py') 
parser.add_argument('--lr_pre' , type=float, default=1e-3)
parser.add_argument('--lr_fine', type=float, default=1e-4)

parser.add_argument('--split', type=float, default=0.8)



def main(args):

    #SugModels = args.out   # suggesting models output list
    num_model = args.num_model
    SugParams = args.insp
    args.result_root = os.path.expanduser(args.result_root)

    # create a directory where results will be saved (if necessary)
    if os.path.exists(args.result_root) == False:
        os.makedirs(args.result_root)
    
    ### execute suggest parameter
    sugpfile = './'+ args.insp              # Suggested parameters file from command ...  
    exec(open(sugpfile).read(), globals())  # directely 'sourcing' the outside python prog.
    
    ### gen Suggest models command line
    outf_SugM = './' + args.sh
    OUTSUGM = open( outf_SugM, 'a' )
    
    
    import csv

    csv_data1 = pd.read_csv( './'+args.incsvone )

    dfone = pd.DataFrame(csv_data1)

    csv_data2 = pd.read_csv('./'+args.incsvtwo)

    dftwo = pd.DataFrame(csv_data2)
    print( dfone )
    print( dftwo )

    del dfone['Time']
    del dfone['Date']
    dfone['Time'] =  dftwo['Time'] 
    
    seq = sorted(dfone.Acc, reverse=True)
    dfone['Rank'] = [1+seq.index(v) for v in dfone.Acc]
    
    dfone['Acc2'] =  dftwo['Acc'] 

    seq = sorted(dfone.Acc2, reverse=True)
    dfone['Rank2'] = [1+seq.index(v) for v in dfone.Acc2]

    print( dfone )
    
    
    
    dfone.to_csv( './'+args.csv , index=False)
    


    df1 = dftwo.sort_values(by=['Acc'], ascending=False)  #; display(df1.head(5))
    df2 = dftwo.sort_values(by=['Models', 'Acc'], ascending=False)  #; display(df2.head(5))
    
    
    for z in range( len(df1['Acc'].tolist()) ):
        print("# %g " % ( df1['Acc'].tolist()[z] ) , file = OUTSUGM )
        print("python3 Train_Model_Eval.py --ind %s --insp %s --mod %s  --epoch %d  --batchsize %d  --optimizer %s --top %s --out_h5 True --log %s --csv %s   " %
              ( args.ind, args.insp, df1['Models'].tolist()[z], df1['Epoch'].tolist()[z], df1['BatSz'].tolist()[z], df1['Optim'] .tolist()[z], df1['Top'] .tolist()[z], args.log, args.csv ) , file = OUTSUGM  )
         

    OUTSUGM.close()
    


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


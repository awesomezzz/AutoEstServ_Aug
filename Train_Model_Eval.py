from keras.applications.xception import Xception as xcmodel, preprocess_input as xcprep
from keras.applications.mobilenetv2 import MobileNetV2 as mv2model, preprocess_input as mv2prep
from keras.applications.nasnet import NASNetMobile as nasmodel, preprocess_input as nasmprep
from keras.applications.resnet50 import ResNet50 as resmodel, preprocess_input as resprep
from keras.applications.densenet import DenseNet169 as dsmodel, preprocess_input as dsprep



from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing import image
from keras.losses import categorical_crossentropy
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
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

#进行配置，使用%的GPU 
config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 1.0
session = tf.Session(config=config) 
# 设置session 
KTF.set_session(session )

Opt = { # optimizer function dictionary
    'Adam' : Adam ,
    'RMSprop' : RMSprop , 
    'SGD' : SGD
}


current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()

# defining corresponding input/output images directories
parser.add_argument('--ind', default='IMG_SRC_AUG') 
parser.add_argument('--classes', '-c', default='./classes.txt') 
parser.add_argument('--result_root', '-r',  default='./result/')
parser.add_argument('--out_h5' , default=False)
parser.add_argument('--out_h5_name' , default='Final') 
parser.add_argument('--log' , default='Final_rpt.log') 
parser.add_argument('--csv' , default='FinANA.csv')

# setting learning models parameters
parser.add_argument('--model')
parser.add_argument('--epoch')
parser.add_argument('--batchsize')
parser.add_argument('--optimizer' , default='Adam')
parser.add_argument('--top')
parser.add_argument('--insp' , default='FinParams.py') 
parser.add_argument('--lr_pre' , type=float, default=1e-3)
parser.add_argument('--lr_fine', type=float, default=1e-4)

parser.add_argument('--split', type=float, default=0.8)



def generate_from_paths_and_labels(select_model, preproc_img, TS, input_paths, labels, batch_size):

    num_samples = len(input_paths)
    while 1:
        perm = np.random.permutation(num_samples)
        input_paths = input_paths[perm]
        labels = labels[perm]
        for i in range(0, num_samples, batch_size):
            inputs = list(map(
                lambda x: image.load_img(x, target_size=(TS,TS)),
                input_paths[i:i+batch_size]
            ))
            inputs = np.array(list(map(
                lambda x: image.img_to_array(x),
                inputs
            )))
            inputs = preproc_img(inputs)
            yield (inputs, labels[i:i+batch_size])


def main(args):

    select_ds = args.ind  # input img data directory
    select_ct =  args.classes
    args.result_root = os.path.expanduser(args.result_root)

    # load class names
    with open(select_ct, 'r') as ct:
        classes = ct.readlines()
        classes = list(map(lambda x: x.strip(), classes))
    num_classes = len(classes)

    # make input_paths and labels
    input_paths, labels = [], []
    for class_name in os.listdir(select_ds):
        class_root = os.path.join(select_ds, class_name)
        class_id = classes.index(class_name)
        for path in os.listdir(class_root):
            path = os.path.join(class_root, path)
            if imghdr.what(path) == None:
                # this is not an image file
                continue
            input_paths.append(path)
            labels.append(class_id)

    # convert to one-hot-vector format
    labels = to_categorical(labels, num_classes=num_classes)

    # convert to numpy array
    input_paths = np.array(input_paths)

    # shuffle dataset
    perm = np.random.permutation(len(input_paths))
    labels = labels[perm]
    input_paths = input_paths[perm]

    # split dataset for training and validation
    border = int(len(input_paths) * args.split)
    train_labels, val_labels = labels[:border], labels[border:]
    train_input_paths, val_input_paths = input_paths[:border], input_paths[border:]
    print("Training on %d images and labels" % (len(train_input_paths)))
    print("Validation on %d images and labels" % (len(val_input_paths)))

    # create a directory where results will be saved (if necessary)
    if os.path.exists(args.result_root) == False:
        os.makedirs(args.result_root)


    from keras.models import Model
    
    
    outf_path = args.result_root
    outf_path2 = outf_path + args.log 
    OUTF = open( outf_path2, 'a' )
    
    
    sugpfile = './'+ args.insp              # Suggested parameters file from command ...  
    exec(open(sugpfile).read(), globals())  # directely 'sourcing' the outside python prog.
    
  
    (select_model, preproc_img, TS) = PrepSZ[args.model]
    pretrained_model = select_model(include_top=False, weights='imagenet', input_shape=(TS,TS,3))
    x = pretrained_model.output
    x = GlobalAveragePooling2D()(x)
                                                            
    PreTp = {
        '1layer'   : Dense(1024, activation='relu')(x) , 
        '2layer_1' : Dense(31, activation='relu')(Dense(1024, activation='relu')(x)),
        '2layer_2' : Dense(45, activation='relu')(Dense(1024, activation='relu')(x)) , 
        '2layer_3' : Dense(63, activation='relu')(Dense(1024, activation='relu')(x)) ,
        '3layer'   : Dense(16, activation='relu')(Dense(128, activation='relu')(Dense(1024, activation='relu')(x)))
    }
   
    (select_epoch , select_bs, select_opt, select_tp) = (int(args.epoch), int(args.batchsize), Opt[args.optimizer], PreTp[args.top]) 
                           
    x = select_tp   # contruct the top layer of the machine code..
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=pretrained_model.inputs, outputs=predictions)
    
    print(type(args.epoch))
    print('model build and start training(freeze base) ...  '  )

    print('# *** [TrainRpt(%s, %s, %s, %s, %s)] ' % (args.model,args.epoch,args.batchsize,args.optimizer,args.top), file=OUTF )
    print ('#BEGIN: ',time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()) , file=OUTF) 

    start = time.time()
        
    # compile model
    model.compile(
        loss      = categorical_crossentropy,
        optimizer = select_opt(lr=args.lr_pre),
        metrics   = ['accuracy']
    )

    # train
    hist_pre = model.fit_generator(
        generator       = generate_from_paths_and_labels(
            select_model, preproc_img, TS,  #m,
            input_paths = train_input_paths,
            labels      = train_labels,
            batch_size  = select_bs
        ),
        steps_per_epoch = math.ceil(len(train_input_paths) / select_bs),
        epochs          = select_epoch,
        validation_data = generate_from_paths_and_labels(
            select_model, preproc_img, TS,  #m,
            input_paths = val_input_paths,
            labels      = val_labels,
            batch_size  = select_bs
        ),
        validation_steps= math.ceil(len(val_input_paths) / select_bs),
        verbose         = 1 ,
    )
    
    end1 = time.time()
        

    print('model build and start training(tain all) ...  '  )

    # recompile
    model.compile(
        optimizer=select_opt(lr=args.lr_fine),
        loss=categorical_crossentropy,
        metrics=['accuracy']
    )

    # train
    hist_fine = model.fit_generator(
        generator       = generate_from_paths_and_labels(
            select_model, preproc_img, TS,  #m,
            input_paths = train_input_paths,
            labels      = train_labels,
            batch_size  = select_bs
        ),
        steps_per_epoch = math.ceil(len(train_input_paths) / select_bs),
        epochs          = select_epoch,
        validation_data = generate_from_paths_and_labels(
            select_model, preproc_img, TS,  #m,
            input_paths = val_input_paths,
            labels      = val_labels,
            batch_size  = select_bs
        ),
        validation_steps= math.ceil(len(val_input_paths) / select_bs),
        verbose         = 1,
    )

    print('hist_fine : ',hist_fine)
        
    end = time.time()
    
    spt = '(spent %.1f secs)'% (end-start)
    spt0 = (end1 -start)
    spt1 = (end - end1)
    spt2 = (end-start)


    #(m,e,b,o,t) = (M[args.model],E[args.epoch],B[args.batchsize],O[args.optimizer],T[args.top])
    m = Models.index(args.model)
    e = Epoch.index(int(args.epoch))
    b = BatSz.index(int(args.batchsize))
    o = Optim.index(args.optimizer)
    t = Top.index(args.top)
    
    print ('#END:   ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()), spt, file=OUTF )
    
    print ('time_spt0 [%d][%d][%d][%d][%d] = %g' %  (m,e,b,o,t, spt0), file=OUTF )
    print ('time_spt1 [%d][%d][%d][%d][%d] = %g' %  (m,e,b,o,t, spt1), file=OUTF )
    print ('time_spt2 [%d][%d][%d][%d][%d] = %g' %  (m,e,b,o,t, spt2), file=OUTF )
                    

    #plt_val_loss(m, e, b, o, t)

    val_acc = hist_pre.history['val_acc']
    val_acc.extend(hist_fine.history['val_acc'])
    val_acc_all  = hist_fine.history['val_acc']

    #save after three decimal piont and convert type

    val_acc_all  = np.around(val_acc_all, decimals=3)
    val_acc_all  = val_acc_all.tolist()
 

    print ('val_acc  [%d][%d][%d][%d][%d] = %s' % (m,e,b,o,t, val_acc_all), file = OUTF)
    print ('final_acc[%d][%d][%d][%d][%d] = %g' % (m,e,b,o,t, val_acc_all[-1]), file = OUTF)


    srccsv = './'+ args.csv
    df = pd.read_csv( srccsv )
    df.loc[len(df)] = [ args.model, args.epoch, args.batchsize, args.optimizer, args.top, val_acc_all[-1], spt2, time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())  ]
    df.to_csv('./'+ args.csv, index=False)
    
  #  print ('DFL.append( [\'%s\', %s, %s, \'%s\', \'%s\', %g, %g ] )' % 
  #         (args.model, args.epoch, args.batchsize, args.optimizer, args.top, val_acc_all[-1], spt2), file = OUTF)
    
    
    #from distutils.util import strtobool
    modsave = args.out_h5
    #modsave = strtobool(modsave)
    
    if modsave == "True" :
        model.save(os.path.join(args.result_root, args.model+'_ep_'+args.epoch+'_bs_'+args.batchsize+'_'+args.optimizer+'_'+args.top+'_'+ args.out_h5_name +'.h5'))

    
    
    K.clear_session()
    

    
    OUTF.close()
    



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


from keras.applications.xception import Xception as xcmodel, preprocess_input as xcprep
from keras.applications.mobilenetv2 import MobileNetV2 as mv2model, preprocess_input as mv2prep
from keras.applications.nasnet import NASNetMobile as nasmodel, preprocess_input as nasmprep
from keras.applications.resnet50 import ResNet50 as resmodel, preprocess_input as resprep
from keras.applications.densenet import DenseNet169 as dsmodel, preprocess_input as dsprep

import math, os, time, datetime
import argparse
import itertools
import pandas as pd


current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()

# defining corresponding input/output images directories
parser.add_argument('--ind', default='IMG_SRC_AUG_SHK') 
parser.add_argument('--sh' , default='SugModels_SHK.sh')
parser.add_argument('--log' , default='SugModels_SHK.log')
parser.add_argument('--csv' , default='PreANA.csv')
parser.add_argument('--classes', '-c', default='./classes.txt') 
parser.add_argument('--result_root', '-r',  default='./result/')

# setting learning models parameters
parser.add_argument('--insp' , default='PreParams.py') 
parser.add_argument('--lr_pre' , type=float, default=1e-3)
parser.add_argument('--lr_fine', type=float, default=1e-4)

parser.add_argument('--split', type=float, default=0.8)


def main(args):

    ### execute suggest parameter
    sugpfile = './'+ args.insp              # Suggested parameters file from command ...  
    exec(open(sugpfile).read(), globals())  # directely 'sourcing' the outside python prog.
    
    ### gen Suggest models command line
    outf_SugM = './' + args.sh
    OUTSUGM = open( outf_SugM, 'a' )

    #print("cd ..", file = OUTSUGM)

    
    # Here we show how to enumerate all possible parameter lists contents ...
    lenpl  = [ len(ParList[x]) for x in range(len(ParList)) ]
    ranges = [ range(x) for x in lenpl]
    Alist  = [ x for x in itertools.product(*ranges) ]

    
  #  df = pd.DataFrame( columns=ParNames)
  #  i = 0
    for idx in Alist:
        m, e, b, o, t = idx
        print("python3 Train_Model_Eval.py --ind %s --insp %s --mod %s  --epoch %d  --batchsize %d  --optimizer %s --top %s --out_h5 False --log %s --csv %s " % 
              ( args.ind, args.insp, Models[m], Epoch[e], BatSz[b], Optim[o], Top[t], args.log, args.csv ) , file = OUTSUGM )
        #df.loc[i] = [Models[m], Epoch[e], BatSz[b], Optim[o], Top[t], rand.uniform(0.5, 1), rand.uniform(30, 100)]
    #    df.loc[i] = [Models[m], Epoch[e], BatSz[b], Optim[o], Top[t]]
    #    i += 1

    #df.to_csv(args.csv, index=False) 

    OUTSUGM.close()
    


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


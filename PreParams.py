####  Prelimary run Parameters set


Models       = [ 'ResNet50', 'MobileNetV2', 'Xception', 'NasNetMobile', 'DenseNet169' ]
Epoch        = [ 7 ]
BatSz        = [ 15 ]
Optim        = [ 'Adam' ]
Top          = [ '1layer', '2layer_1', '2layer_2', '2layer_3', '3layer' ]


ParList  = [ Models, Epoch, BatSz, Optim, Top ]
ParNames = [ 'Models', 'Epoch', 'BatSz', 'Optim', 'Top' ]

PrepSZ = {
    'ResNet50'     : (resmodel, resprep, 224),
    'MobileNetV2'   : (mv2model, mv2prep, 224),
    'Xception'     : (xcmodel, xcprep, 299), 
    'NasNetMobile' : (nasmodel, nasmprep, 224), 
    'DenseNet169'  : (dsmodel, dsprep, 224)
}

# python3 Train_Model_Eval.py --ind IMG_SHK --insp PreParams.py --mod ResNet50  --epoch 1  --batchsize 15  --optimizer Adam --top 1layer --out_h5 False --outrpt xxx.out
#python3 SuggestFinal.py --ind IMG_SRC --insp FinParams.py --inshlog xxx.log --incsv PreANA.csv  --num_model 5 --sh xxx_SRC.sh --csv SugANA.csv


# python3 GenPreMod.py --ind IMG_SHK --insp PreParams.py --out SugModels_SHK.sh --outrpt SugModels_SHK.out

# python3 SuggestFinal.py --ind IMG_SRC --insp FinParams.py --inshrpt SugModels_SHK.out  --num_model 5 --out SugModels_SRC.sh

# python3 SuggestFinal.py --ind IMG_SRC --insp FinParams.py --inshrpt SuggestFinal_SRC.out  --num_model 5 --out SugModels_Fin.sh



# python3 GenPreMod.py --ind IMG_SHK --insp PreParams.py --out SugModels_SHK.sh --outrpt SugModels_SHK.out
# sh SugModels_SHK.sh 
# python3 SuggestFinal.py --ind IMG_SHK --insp PreParams.py --inshrpt SugModels_SHK.out  --num_model 5 --out SHK.sh


# python3 GenPreMod.py --ind IMG_SRC --insp PreParams.py --out SugModels_SRC.sh --outrpt SugModels_SRC.out
# sh SugModels_SRC.sh 
# python3 SuggestFinal.py --ind IMG_SRC --insp PreParams.py --inshrpt SugModels_SHK.out  --num_model 5 --out SRC.sh


#python3 SuggestFinal.py --ind ${SRCIMGD} --insp ${FINPAR} --inshrpt ${SHKRPT} --num_model 5 --out ${SUGMODSRC}
### *****
# python3 GenPreMod.py --ind IMG_SHK --insp PreParams.py --out SugModels_SHK.sh --outrpt SugModels_SHK.out
# sh SugModels_SHK.sh 
#python3 SuggestFinal.py --ind IMG_SRC --insp FinParams.py --inshrpt SugModels_SHK.out  --num_model 5 --out SugModels_SRC.sh
# sh SugModels_SRC.sh 
#python3 SuggestFinal_obs.py --ind IMG_SRC --insp FinParams.py --inshrpt SugModels_SRC.out  --num_model 5 --out SugModels_FIN.sh





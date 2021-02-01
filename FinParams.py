####  Final run Parameters set


Models       = [ 'ResNet50', 'MobileNetV2', 'Xception', 'NasNetMobile', 'DenseNet169' ]
Epoch        = [ 20 ]
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

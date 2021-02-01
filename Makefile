SRCIMGD = IMG_SRC
SRCAUG  = IMG_SRC_AUG
IMGD    = IMG_SRC_AUG_SHK
TIMG    = IMG/crocus/image_0323.jpg
MODEL   = result/model_pre_ep1_valloss0.754.h5
PREPAR  = PreParams.py
FINPAR  = FinParams.py
SUGMODSHK  = PreANA_SHK.sh
SUGMODSRC  = SugANAs_SRC.sh
SHKRPT  = PreANA_SHK.log
FINRPT  = SugANA_SRC.log
PRECSV  = PreANA.csv
SHKCSV  = SHKANA.csv
SUGCSV  = SugANA.csv
FINANA  = FinANA.csv
FINSUGMODSRC = OBS_FIN.sh

GOAL: newcsv ${SRCAUG} ${IMGD} ${SUGMODSHK} ${SHKRPT} ${SUGMODSRC} ${FINRPT} ${FINSUGMODSRC}

${SRCAUG}: 
	python3 ExecAug.py --ind ${SRCIMGD} --outd ${SRCAUG}

${IMGD}: 
	python3 Shrink_data.py --ind ${SRCAUG} --out ${IMGD} --ratio 0.01 --ran_seed 5

${SUGMODSHK}:   # Evaluating and Producing Suitable Model Parameters from shrunk ${IMGD} dir...
	python3 GenPreMod.py --ind ${IMGD} --insp ${PREPAR} --sh ${SUGMODSHK} --log ${SHKRPT} --csv ${PRECSV}
    
${SHKRPT}: 
	sh ${SUGMODSHK}

${SUGMODSRC}:   # Evaluating all posssible models and produce the final report.
	python3 SuggestFinal.py --ind ${SRCAUG} --insp ${FINPAR} --inshlog ${SHKRPT} --incsv ${PRECSV} --num_model 5 --sh ${SUGMODSRC} --log ${FINRPT} --csv ${SHKCSV} --csvcmd ${SUGCSV}
    
${FINRPT}: 
	sh ${SUGMODSRC}
    
${FINSUGMODSRC}: 
	python3 ObsFinal.py --ind ${SRCIMGD} --insp ${FINPAR} --inshlog ${FINRPT} --incsvone SHKANA.csv --incsvtwo ${SUGCSV} --sh ${FINSUGMODSRC} --csv ${FINANA}
    

    
wewant: 
	echo "python3 Gen_model_retrain.py ${SRCIMGD} classes.txt ./result2  --mod Xcept --top [top_layer_opts] --epochs_pre ${EPP} --epochs_fine 10 --lr_fine 5e-4 --out model1.h5"
	#python3 gen_model_retrain-2.py ${SRCIMGD} --mod MobilNetV2  --epoch 25  --batchsize 15  --optimizer Adam --top 2layer_2 --out_h5 True --out_finrpt ${FINRPT}


newcsv:
	cp ModsEval-head.csv PreANA.csv 
	cp ModsEval-head.csv SugANA.csv 
	cp ModsEval-head.csv SHKANA.csv 
	cp ModsEval-head.csv FinANA.csv 

k: ${IMGD} 
	\rm -rf ${IMGD}
    
    
    

classes.txt:
	ls ${SRCIMGD} > classes.txt

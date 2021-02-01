


py0: ExecAug.py
py1: GenPreMod.py
py2: SuggestFinal.py
py3: Train_Model_Eval.py
py4: ObsFinal.py

sh1: PreANA_SHK.sh
sh2: SugANAs_SRC.sh

log1: PreANA_SHK.log
log2: SugANA_SRC.log


py0 ---> Shrink_data.py  --->


              run SHK data                     get SHK no.5
          ----------------------               -------------
          |                    |               |           |
py1 ---> sh1 ---> py3---> log1 & Pre.csv ---> py2 ---> sh2 & SHK.csv ---> py3---> log2 & Sug.csv & h5 ---> py4---> sh3 & Fin.csv 
                             (全組合)                 (run SHK後前幾名組合)         (run SRC後前幾名組合)           (結合前兩個csv做比較)
                                                           |                          |                     |          |
                                                           ----------------------------                     ------------
                                                                   run SRC data                             get SRC no.5

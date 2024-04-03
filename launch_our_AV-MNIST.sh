for i in $(seq 0 1 4)
do
    for k in CONTRA #ORTO
    do
        python main_cs_v3.py AV-MNIST MNIST SPECTRO $i $k
        python main_cs_v3.py AV-MNIST SPECTRO MNIST $i $k
        #python main_cs.py SUNRGBD RGB DEPTH $i $k
        #python main_cs.py SUNRGBD DEPTH RGB $i $k
    done
done
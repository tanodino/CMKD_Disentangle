for i in $(seq 0 1 4)
do
    for k in CONTRA #ORTO
    do
        python main_cs_v4.py AV-MNIST MNIST SPECTRO $i $k > log_v4_AV-MNIST_MNIST_SPECTRO_${i}_${k}
        python main_cs_v4.py AV-MNIST SPECTRO MNIST $i $k > log_v4_AV-MNIST_SPECTRO_MNIST_${i}_${k}
        #python main_cs.py TRISTAR DEPTH THERMAL $i $k
        #python main_cs.py TRISTAR THERMAL DEPTH $i $k
    done
done

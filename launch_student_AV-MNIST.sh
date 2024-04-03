for i in $(seq 0 1 4)
do
    for j in MNIST SPECTRO
    do
        for k in KD1 KD2 DKD MLKD CTKD
        do
            for v in 0 1
            do
                python student.py AV-MNIST MNIST SPECTRO SUM $j $k $v $i
            done
        done
    done
done
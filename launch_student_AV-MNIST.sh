for i in $(seq 0 1 9)
do
    for j in SPECTRO #MNIST
    do
        for k in KD DKD MLKD
        do
            python student.py AV-MNIST MNIST SPECTRO SUM $j $k $i
        done
    done
done
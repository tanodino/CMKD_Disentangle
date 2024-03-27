for i in $(seq 0 1 4)
do
    for j in MS SAR
    do
        for k in KD1 KD2 DKD MLKD CTKD
        do
            for v in 0 1
            do
                python student.py EUROSAT MS SAR SUM $j $k $v $i
            done
        done
    done
done
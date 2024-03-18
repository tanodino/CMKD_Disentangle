for i in $(seq 0 1 9)
do
    for j in MS SAR
    do
        for k in KD DKD MLKD
        do
            python student.py EUROSAT MS SAR SUM $j $k $i
        done
    done
done
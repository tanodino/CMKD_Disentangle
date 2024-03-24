for i in $(seq 0 1 9)
do
    for j in DEPTH THERMAL
    do
        for k in KD DKD MLKD
        do
            python student.py TRISTAR DEPTH THERMAL SUM $j $k $i
        done
    done
done
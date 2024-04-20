for i in $(seq 0 1 4)
do
    for k in KD1 KD2 DKD MLKD CTKD
        do
            for v in 0 1
            do
                python student.py $1 $2 $3 $k $v $i
            done
        done
    done
done
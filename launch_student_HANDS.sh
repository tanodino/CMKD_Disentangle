for i in $(seq 0 1 9)
do
    for j in RGB DEPTH
    do
        for k in KD DKD MLKD
        do
            python student.py HANDS RGB DEPTH SUM $j $k $i
        done
    done
done
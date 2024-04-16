for i in $(seq 0 1 4)
do
    for j in $(seq 1 1 4)
    do
        python main_abla.py $1 $2 $3 $i $j > log_abla_${1}_${2}_${3}_${i}_ABLA${j}
    done
done

# launch_our_v6.sh SUNRGBD RGB DEPTH
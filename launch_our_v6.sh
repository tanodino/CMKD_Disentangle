for i in $(seq 0 1 4)
do
    python main_v6.py $1 $2 $3 $i ORTHO > log_v6_${1}_${2}_${3}_${i}
done

# launch_our_v6.sh SUNRGBD RGB DEPTH
for i in $(seq 0 1 4)
do
    python main_cv_v4_compact.py $1 $2 $3 $i ORTHO > log_v5_${1}_${2}_${3}_${i}
    #python teacher.py $1 $2 $3 $i CONCAT > log_${1}_${2}_${3}_${i}_CONCAT
    #python teacher.py $1 $2 $3 $i SUM > log_${1}_${2}_${3}_${i}_SUM
done

# launch_our_v5.sh SUNRGBD RGB DEPTH
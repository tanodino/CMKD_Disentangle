for i in $(seq 0 1 9)
do
    #python teacher.py $1 $2 $3 $i CONCAT > log_${1}_${2}_${3}_${i}_CONCAT
    python teacher.py $1 $2 $3 $i SUM > log_${1}_${2}_${3}_${i}_SUM
done


#nohup sh launch_teacher.sh EUROSAT MS SAR &
#nohup sh launch_teacher.sh SUNRGBD RGB DEPTH &
#nohup sh launch_teacher.sh AV-MNIST mnist spectro &


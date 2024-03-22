for i in $(seq 0 1 9)
do
    python monoSource.py $1 $2 $i > log_${1}_${2}_${i}
done

#nohup sh launch_monoSource.sh EUROSAT MS &
#nohup sh launch_monoSource.sh EUROSAT SAR &
#nohup sh launch_monoSource.sh SUNRGBD RGB &
#nohup sh launch_monoSource.sh SUNRGBD DEPTH &
#nohup sh launch_monoSource.sh AV-MNIST SPECTRO &
#nohup sh launch_monoSource.sh AV-MNIST MNIST &


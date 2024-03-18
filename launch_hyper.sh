for i in $(seq 0 1 9)
do
    python teacherHYPER.py PAVIA_UNIVERSITY FULL $i > log_PAVIA_UNIVERSITY_FULL_${i}
    python monoSource.py PAVIA_UNIVERSITY HALF $i > log_PAVIA_UNIVERSITY_HALF_${i}
done

#nohup sh launch_monoSource.sh EUROSAT MS &
#nohup sh launch_monoSource.sh EUROSAT SAR &
#nohup sh launch_monoSource.sh SUNRGBD RGB &
#nohup sh launch_monoSource.sh SUNRGBD DEPTH &


for i in $(seq 0 1 4)
do
    for j in $(seq 1 1 4)
    do
        python main_abla.py EUROSAT MS SAR $i $j > log_abla_EUROSAT_MS_SAR_${i}_ABLA${j}
    done
done

# launch_our_v6.sh SUNRGBD RGB DEPTH
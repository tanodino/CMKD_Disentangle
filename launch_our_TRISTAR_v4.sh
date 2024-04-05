for i in $(seq 0 1 4)
do
    for k in CONTRA #ORTO
    do
        python main_cs_v4.py TRISTAR DEPTH THERMAL $i $k > log_v4_TRISTAR_DEPTH_THERMAL_${i}_${k}
        python main_cs_v4.py TRISTAR THERMAL DEPTH $i $k > log_v4_TRISTAR_THERMAL_DEPTH_${i}_${k}
        #python main_cs.py TRISTAR DEPTH THERMAL $i $k
        #python main_cs.py TRISTAR THERMAL DEPTH $i $k
    done
done

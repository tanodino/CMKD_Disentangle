for i in $(seq 0 1 4)
do
    for k in CONTRA #ORTO
    do
        python main_cs_v3_ema.py TRISTAR DEPTH THERMAL $i $k > log_TRISTAR_OUR_DEPTH_THERMAL_${i}_${k}_EMA
        python main_cs_v3_ema.py TRISTAR THERMAL DEPTH $i $k > log_TRISTAR_OUR_THERMAL_DEPTH_${i}_${k}_EMA
        #python main_cs.py TRISTAR DEPTH THERMAL $i $k
        #python main_cs.py TRISTAR THERMAL DEPTH $i $k
    done
done
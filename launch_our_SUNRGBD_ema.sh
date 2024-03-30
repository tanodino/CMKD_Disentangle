for i in $(seq 0 1 4)
do
    for k in CONTRA #ORTO
    do
        python main_cs_v3_ema.py SUNRGBD RGB DEPTH $i $k > log_SUNRGBD_OUR_RGB_DEPTH_${i}_${k}_EMA
        python main_cs_v3_ema.py SUNRGBD DEPTH RGB $i $k > log_SUNRGBD_OUR_DEPTH_RGB_${i}_${k}_EMA
        #python main_cs.py SUNRGBD RGB DEPTH $i $k
        #python main_cs.py SUNRGBD DEPTH RGB $i $k
    done
done
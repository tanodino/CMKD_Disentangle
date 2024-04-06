for i in $(seq 0 1 4)
do
    for k in CONTRA #ORTO
    do
        python main_cs_v4.py SUNRGBD RGB DEPTH $i $k > log_v4_SUNRGBD_RGB_DEPTH_${i}_${k}
        python main_cs_v4.py SUNRGBD DEPTH RGB $i $k > log_v4_SUNRGBD_DEPTH_RGB_${i}_${k}
        #python main_cs.py SUNRGBD RGB DEPTH $i $k
        #python main_cs.py SUNRGBD DEPTH RGB $i $k
    done
done
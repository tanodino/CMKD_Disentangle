for i in $(seq 0 1 4)
do
    for k in CONTRA #ORTO
    do
        python main_cs_v4.py EUROSAT MS SAR $i $k > log_v4_EUROSAT_MS_SAR_${i}_${k}
        python main_cs_v4.py EUROSAT SAR MS $i $k > log_v4_EUROSAT_SAR_MS_${k}_${i}
        #python main_cs.py EUROSAT MS SAR $i $k
        #python main_cs.py EUROSAT SAR MS $i $k
    done
done

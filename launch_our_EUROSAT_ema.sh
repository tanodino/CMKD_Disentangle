for i in $(seq 0 1 4)
do
    for k in CONTRA #ORTO
    do
        python main_cs_v3_ema.py EUROSAT MS SAR $i $k > log_EUROSAT_OUR_MS_SAR_${i}_${k}_EMA
        python main_cs_v3_ema.py EUROSAT SAR MS $i $k > log_EUROSAT_OUR_SAR_MS_${i}_${k}_EMA
        #python main_cs.py EUROSAT MS SAR $i $k
        #python main_cs.py EUROSAT SAR MS $i $k
    done
done
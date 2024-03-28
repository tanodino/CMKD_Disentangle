for i in $(seq 0 1 4)
do
    for k in CONTRA #ORTO
    do
        python main_cs_v3.py EUROSAT MS SAR $i $k
        python main_cs_v3.py EUROSAT SAR MS $i $k
        #python main_cs.py EUROSAT MS SAR $i $k
        #python main_cs.py EUROSAT SAR MS $i $k
    done
done
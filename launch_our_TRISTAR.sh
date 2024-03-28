for i in $(seq 0 1 4)
do
    for k in CONTRA #ORTO
    do
        python main_cs_v3.py TRISTAR DEPTH THERMAL $i $k
        python main_cs_v3.py TRISTAR THERMAL DEPTH $i $k
        #python main_cs.py TRISTAR DEPTH THERMAL $i $k
        #python main_cs.py TRISTAR THERMAL DEPTH $i $k
    done
done
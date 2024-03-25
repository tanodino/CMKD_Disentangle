for i in $(seq 0 1 9)
do
    for k in CONTRA ORTO
    do
        python main_cmkd_v2.py TRISTAR DEPTH THERMAL $i $k
        python main_cmkd_v2.py TRISTAR THERMAL DEPTH $i $k
    done
done
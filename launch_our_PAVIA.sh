for i in $(seq 0 1 9)
do
    for k in CONTRA ORTO
    do
        python main_cs.py PAVIA_UNIVERSITY HALF FULL $i $k
    done
done

for i in $(seq 0 1 9)
do
    for k in CONTRA ORTO
    do
        python main_cs.py SUNRGBD RGB DEPTH $i $k
        python main_cs.py SUNRGBD DEPTH RGB $i $k
    done
done
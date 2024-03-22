for i in $(seq 0 1 9)
do
    for k in CONTRA ORTO
    do
        python main_cs.py HANDS RGB DEPTH $i $k
        python main_cs.py HANDS DEPTH RGB $i $k
    done
done
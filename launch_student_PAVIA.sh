for i in $(seq 0 1 9)
do
    python student.py PAVIA_UNIVERSITY FULL FULL FULL HALF KD $i
    python student.py PAVIA_UNIVERSITY FULL FULL FULL HALF DKD $i
    python student.py PAVIA_UNIVERSITY FULL FULL FULL HALF MLKD $i
done
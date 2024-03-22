for i in $(seq 0 1 9)
do
    for k in CONTRA ORTO
    do
        python main_cs.py AV-MNIST MNIST SPECTRO $i $k
        python main_cs.py AV-MNIST SPECTRO MNIST $i $k
    done
done


# https://www.sciencedirect.com/science/article/pii/S2352340921000755
# HANDS: an RGB-D dataset of static hand-gestures for human-robot interaction
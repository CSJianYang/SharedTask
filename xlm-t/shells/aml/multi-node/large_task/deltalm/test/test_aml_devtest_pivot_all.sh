export PYTHONWARNINGS="ignore"

dir=$1
MODEL=$2
batchsize=$3
beam=$4
TEST_PIVOT_SHELL=./shells/aml/multi-node/large_task/deltalm/test/test_aml_devtest_pivot.sh
TEST_DIRECT_SHELL=./shells/aml/multi-node/large_task/deltalm/test/test_aml_devtest.sh
echo $dir
if [ "$dir" == "e2x" ]; then
    src=en
    TGTS=(af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu)
    for tgt in "${TGTS[@]}"; do
        echo "${src}->${tgt}"
        bash $TEST_DIRECT_SHELL $src $tgt $batchsize $beam $MODEL
    done
elif [ "$dir" == "x2e" ]; then
    SRCS=(af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu)
    tgt=en
    for src in "${SRCS[@]}"; do
        echo "${src}->${tgt}"
        bash $TEST_DIRECT_SHELL $src $tgt $batchsize $beam $MODEL
    done
elif [ "$dir" == "2x" ]; then
    src=$5
    TGTS=(af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu)
    for tgt in "${TGTS[@]}"; do
        if [ "$src" != "$tgt" ]; then
            if [ "$src" == "en" -o "$tgt" == "en" ]; then
                echo "${src}->${tgt}"
                bash $TEST_DIRECT_SHELL $src $tgt $batchsize $beam $MODEL
            else
                if [ "$tgt" == "af" ]; then
                    echo "${src}->${tgt}: Overwrite the ${src}->en"
                    bash $TEST_PIVOT_SHELL $src $tgt $batchsize $beam $MODEL "Yes"
                else
                    echo "${src}->${tgt}: Will not Overwrite the ${src}->en"
                    bash $TEST_PIVOT_SHELL $src $tgt $batchsize $beam $MODEL "No"
                fi
            fi
        fi
    done
else
    SRCS=(af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu)
    TGTS=(af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu)
    for src in "${SRCS[@]}"; do
        for tgt in "${TGTS[@]}"; do
            if [ "$src" == "en" -o "$tgt" == "en" ]; then
                echo "${src}->${tgt}"
                bash $TEST_DIRECT_SHELL $src $tgt $batchsize $beam $MODEL
            else
                echo "${src}->${tgt}"
                bash $TEST_PIVOT_SHELL $src $tgt $batchsize $beam $MODEL
            fi
        done
    done
fi
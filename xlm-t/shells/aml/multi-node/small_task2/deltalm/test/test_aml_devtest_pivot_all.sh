export PYTHONWARNINGS="ignore"

dir=$1
MODEL=$2
batchsize=$3
beam=$4
TEST_PIVOT_SHELL=./shells/aml/multi-node/small_task2/deltalm/test/test_aml_devtest_pivot.sh
TEST_DIRECT_SHELL=./shells/aml/multi-node/small_task2/deltalm/test/test_aml_devtest.sh
echo $dir
if [ "$dir" == "e2x" ]; then
    src=en
    TGTS=(id jv ms ta tl)
    for tgt in "${TGTS[@]}"; do
        echo "${src}->${tgt}"
        bash $TEST_DIRECT_SHELL $src $tgt $batchsize $beam $MODEL
    done
elif [ "$dir" == "x2e" ]; then
    SRCS=(id jv ms ta tl)
    tgt=en
    for src in "${SRCS[@]}"; do
        echo "${src}->${tgt}"
        bash $TEST_DIRECT_SHELL $src $tgt $batchsize $beam $MODEL
    done
elif [ "$dir" == "2x" ]; then
    src=$5
    TGTS=(en id jv ms ta tl)
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
    SRCS=(en id jv ms ta tl)
    TGTS=(en id jv ms ta tl)
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
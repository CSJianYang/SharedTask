export PYTHONWARNINGS="ignore"

dir=$1
MODEL=$2
batchsize=$3
beam=$4
TEST_SHELL=./shells/aml/multi-node/small_task2/deltalm/test/test_aml_devtest.sh

echo $dir
if [ "$dir" == "e2x" ]; then
    src=en
    TGTS=(id jv ms ta tl)
    for tgt in "${TGTS[@]}"; do
        echo "${src}->${tgt}"
        bash $TEST_SHELL $src $tgt $batchsize $beam $MODEL
    done
elif [ "$dir" == "x2e" ]; then
    SRCS=(id jv ms ta tl)
    tgt=en
    for src in "${SRCS[@]}"; do
        echo "${src}->${tgt}"
        bash $TEST_SHELL $src $tgt $batchsize $beam $MODEL
    done
elif [ "$dir" == "2x" ]; then
    src=$5
    TGTS=(en id jv ms ta tl)
    for tgt in "${TGTS[@]}"; do
        if [ "$src" != "$tgt" ]; then
            echo "${src}->${tgt}"
            bash $TEST_SHELL $src $tgt $batchsize $beam $MODEL
        fi
    done
else
    SRCS=(en id jv ms ta tl)
    TGTS=(en id jv ms ta tl)
    for src in "${SRCS[@]}"; do
        for tgt in "${TGTS[@]}"; do
            if [ "$src" != "$tgt" ]; then
                echo "${src}->${tgt}"
                bash $TEST_SHELL $src $tgt $beam $MODEL
            fi
        done
    done
fi
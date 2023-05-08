export PYTHONWARNINGS="ignore"

dir=$1
MODEL=$2
batchsize=$3
beam=$4
TEST_SHELL=./shells/aml/multi-node/small_task1/test/test_aml_devtest.sh

echo $dir
if [ "$dir" == "e2x" ]; then
    src=en
    TGTS=(et hr sr mk sr)
    for tgt in "${TGTS[@]}"; do
        echo "${src}->${tgt}"
        bash $TEST_SHELL $src $tgt $batchsize $beam $MODEL
    done
elif [ "$dir" == "x2e" ]; then
    SRCS=(et hr sr mk sr)
    tgt=en
    for src in "${SRCS[@]}"; do
        echo "${src}->${tgt}"
        bash $TEST_SHELL $src $tgt $batchsize $beam $MODEL
    done
elif [ "$dir" == "2x" ]; then
    src=$5
    TGTS=(en et hr hu mk sr)
    for tgt in "${TGTS[@]}"; do
        if [ "$src" != "$tgt" ]; then
            echo "${src}->${tgt}"
            bash $TEST_SHELL $src $tgt $batchsize $beam $MODEL
        fi
    done
else
    SRCS=(en et hr hu mk sr)
    TGTS=(en et hr hu mk sr)
    for src in "${SRCS[@]}"; do
        for tgt in "${TGTS[@]}"; do
            if [ "$src" != "$tgt" ]; then
                echo "${src}->${tgt}"
                bash $TEST_SHELL $src $tgt $beam $MODEL
            fi
        done
    done
fi
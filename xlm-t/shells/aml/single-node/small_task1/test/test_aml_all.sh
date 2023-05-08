export PYTHONWARNINGS="ignore"

dir=$1
beam=5
TEST_SHELL=./shells/aml/single-node/small_task1/test/test_aml.sh

MODEL=$2
#MODEL=/mnt/input/SharedTask/large-scale/PretrainedModel/mm100_175M/flores101_mm100_175M/model.pt
#MODEL=/mnt/input/SharedTask/large-scale/PretrainedModel/mm100_615M/flores101_mm100_615M/model.pt
echo $dir
if [ "$dir" == "e2x" ]; then
  src=en
  TGTS=(hr hu et sr mk)
  for tgt in "${TGTS[@]}"; do
      echo "${src}->${tgt}"
      bash $TEST_SHELL $src $tgt $beam $MODEL
  done
elif [ "$dir" == "x2e" ]; then
  SRCS=(hr hu et sr mk)
  tgt=en
  for src in "${SRCS[@]}"; do
      echo "${src}->${tgt}"
      bash $TEST_SHELL $src $tgt $beam $MODEL
  done
else
  src=en
  TGTS=(hr hu et sr mk)
  for tgt in "${TGTS[@]}"; do
      echo "${src}->${tgt}"
      bash $TEST_SHELL $src $tgt $beam $MODEL
  done
  
  SRCS=(hr hu et sr mk)
  tgt=en
  for src in "${SRCS[@]}"; do
      echo "${src}->${tgt}"
      bash $TEST_SHELL $src $tgt $beam $MODEL
  done
fi
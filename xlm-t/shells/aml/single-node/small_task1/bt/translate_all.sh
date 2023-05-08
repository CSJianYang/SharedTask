export PYTHONWARNINGS="ignore"

dir=$1
beam=5
TEST_SHELL=./shells/aml/single-node/small_task1/bt/translate.sh

INPUT=$1
MODEL=$2
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
elif [ "$dir" == "e2x_x2e" ]; then
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
else
  SRCS=(en hr hu et sr mk)
  TGTS=(en hr hu et sr mk)
  for src in "${SRCS[@]}"; do
      for tgt in "${TGTS[@]}"; do
          echo "${src}->${tgt}"
          bash $TEST_SHELL $src $tgt $beam $MODEL
      done
  done
fi
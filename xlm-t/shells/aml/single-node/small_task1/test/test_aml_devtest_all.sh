export PYTHONWARNINGS="ignore"

dir=$1
MODEL=$2
beam=$3
TEST_SHELL=./shells/aml/single-node/small_task1/test/test_aml_devtest.sh

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
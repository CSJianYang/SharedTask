export PYTHONWARNINGS="ignore"

dir=$1
beam=5
TEST_SHELL=./shells/aml/multi-node/small_task1/32GPU/deltalm/test/test_aml.sh

MODEL=$2
#MODEL=/mnt/input/SharedTask/large-scale/PretrainedModel/mm100_175M/flores101_mm100_175M/model.pt
#MODEL=/mnt/input/SharedTask/large-scale/PretrainedModel/mm100_615M/flores101_mm100_615M/model.pt
echo $dir
if [ "$dir" == "e2x" ]; then
  src=en
  TGTS=(af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu)
  for tgt in "${TGTS[@]}"; do
      echo "${src}->${tgt}"
      bash $TEST_SHELL $src $tgt $beam $MODEL
  done
elif [ "$dir" == "x2e" ]; then
  SRCS=(af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu)
  tgt=en
  for src in "${SRCS[@]}"; do
      echo "${src}->${tgt}"
      bash $TEST_SHELL $src $tgt $beam $MODEL
  done
else
  src=en
  TGTS=(af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu)
  for tgt in "${TGTS[@]}"; do
      echo "${src}->${tgt}"
      bash $TEST_SHELL $src $tgt $beam $MODEL
  done
  
  SRCS=(af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu)
  tgt=en
  for src in "${SRCS[@]}"; do
      echo "${src}->${tgt}"
      bash $TEST_SHELL $src $tgt $beam $MODEL
  done
fi
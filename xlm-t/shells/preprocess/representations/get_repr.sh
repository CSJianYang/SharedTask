LANGS="af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu"
GENERATE_REPR=/home/v-jiaya/SharedTask/xlm-t/get_encoder_representation.py
PYTHON=/home/v-jiaya/miniconda3/envs/amlt8/bin/python
DATA=/home/v-jiaya/SharedTask/data/thunder/flores101_dataset/devtest-code_spm/
for lg in ${LANGS[@]}; then
    echo "Get $lg Representations..."
    $PYTHON $GENERATE_REPR --src-fn $DATA/valid.$lg
do
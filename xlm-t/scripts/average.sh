#MODEL_DIR=/mnt/input/SharedTask/thunder/small_task2/Filter_v1/model/24L-12L-A100/
#echo "start averaging checkpoints"
#python average_checkpoints.py --inputs $MODEL_DIR/avg4_14.pt $MODEL_DIR/avg5_14.pt --output $MODEL_DIR/avg.pt
#python average_checkpoints.py --inputs $MODEL_DIR/checkpoint4.pt $MODEL_DIR/checkpoint5.pt $MODEL_DIR/checkpoint6.pt $MODEL_DIR/checkpoint7.pt $MODEL_DIR/checkpoint8.pt  --output $MODEL_DIR/avg4_8.pt
#python average_checkpoints.py --inputs $MODEL_DIR/checkpoint5.pt $MODEL_DIR/checkpoint6.pt $MODEL_DIR/checkpoint7.pt $MODEL_DIR/checkpoint8.pt $MODEL_DIR/checkpoint9.pt  --output $MODEL_DIR/avg5_9.pt




#MODEL_DIR=/mnt/input/SharedTask/large-scale/small_task1/download/model/M2M100-lr1e-4/
#echo "start averaging checkpoints"
#python average_checkpoints.py --inputs $MODEL_DIR --num-epoch-checkpoints=5 --checkpoint-upper-bound=50  --output $MODEL_DIR/avg46_50.pt

#MODEL_DIR=/mnt/input/SharedTask/large-scale/small_task1/download/model/M2M100-lr1e-4/
#echo "start averaging checkpoints"
#python average_checkpoints.py --inputs $MODEL_DIR --num-epoch-checkpoints=10 --checkpoint-upper-bound=50  --output $MODEL_DIR/avg41_50.pt



#MODEL_DIR=/mnt/input/SharedTask/large-scale/small_task1/download/model/M2M100-lr5e-5/
#echo "start averaging checkpoints"
#python average_checkpoints.py --inputs $MODEL_DIR --num-epoch-checkpoints=5 --checkpoint-upper-bound=50  --output $MODEL_DIR/avg46_50.pt

#MODEL_DIR=/mnt/input/SharedTask/large-scale/small_task1/download/model/M2M100-lr5e-5/
#echo "start averaging checkpoints"
#python average_checkpoints.py --inputs $MODEL_DIR --num-epoch-checkpoints=10 --checkpoint-upper-bound=50  --output $MODEL_DIR/avg41_50.pt


#MODEL_DIR=/mnt/input/SharedTask/thunder/small_task2/Filter_v1/model/36L-12L-V100-pivot/
#echo "start averaging checkpoints"
#python average_checkpoints.py --inputs $MODEL_DIR --num-epoch-checkpoints=11 --checkpoint-upper-bound=14  --output $MODEL_DIR/avg4_14.pt


MODEL_DIR=/mnt/input/SharedTask/thunder/large_track/data/Filter_v1/model/36L-12L/
echo "start averaging checkpoints"
python average_checkpoints.py --inputs $MODEL_DIR --num-epoch-checkpoints=15 --checkpoint-upper-bound=40  --output $MODEL_DIR/avg26_40.pt



#MODEL_DIR=/mnt/input/SharedTask/thunder/small_task2/Filter_v1/model/36L-12L-V100/
#echo "start averaging checkpoints"
#python average_checkpoints.py --inputs $MODEL_DIR --num-epoch-checkpoints=10 --checkpoint-upper-bound=15  --output $MODEL_DIR/avg6_15.pt
#MODEL_DIR=/mnt/input/SharedTask/thunder/small_task2/Filter_v1/model/24L-12L-A100/
#echo "start averaging checkpoints"
#python average_checkpoints.py --inputs $MODEL_DIR --num-epoch-checkpoints=5 --checkpoint-upper-bound=10  --output $MODEL_DIR/avg5_10.pt

#MODEL_DIR=/mnt/input/SharedTask/thunder/small_task2/Filter_v1/model/24L-12L-step2/
#echo "start averaging checkpoints"
#python average_checkpoints.py --inputs $MODEL_DIR --num-epoch-checkpoints=9 --checkpoint-upper-bound=18  --output $MODEL_DIR/avg9_18.pt

#MODEL_DIR=/mnt/input/SharedTask/thunder/large_track/data/Filter_v1/model/24L-12L/
#echo "start averaging checkpoints"
#python average_checkpoints.py --inputs $MODEL_DIR --num-epoch-checkpoints=10 --checkpoint-upper-bound=70  --output $MODEL_DIR/avg61_70.pt

#echo "Removing Optimizer $MODEL_DIR/simple_model.pt"
#python ./scripts/remove_optimizer.py
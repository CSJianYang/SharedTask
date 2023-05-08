PYTHON=/home/v-jiaya/anaconda3-fairseq-0.9.0/bin/python
AVERAGE=/home/v-jiaya/DeepMNMT/xlm-t/average_checkpoints.py
MODEL_DIR=/home/v-jiaya/DeepMNMT/models/xlmt-ls-e2x-base/
#$PYTHON $AVERAGE --inputs $MODEL_DIR/checkpoint3.pt $MODEL_DIR/checkpoint5.pt $MODEL_DIR/checkpoint6.pt $MODEL_DIR/checkpoint7.pt $MODEL_DIR/checkpoint7.pt  --output $MODEL_DIR/avg3_7.pt
#$PYTHON $AVERAGE --inputs $MODEL_DIR/checkpoint4.pt $MODEL_DIR/checkpoint5.pt $MODEL_DIR/checkpoint6.pt $MODEL_DIR/checkpoint7.pt $MODEL_DIR/checkpoint8.pt  --output $MODEL_DIR/avg4_8.pt
#$PYTHON $AVERAGE --inputs $MODEL_DIR/checkpoint5.pt $MODEL_DIR/checkpoint6.pt $MODEL_DIR/checkpoint7.pt $MODEL_DIR/checkpoint8.pt $MODEL_DIR/checkpoint9.pt  --output $MODEL_DIR/avg5_9.pt

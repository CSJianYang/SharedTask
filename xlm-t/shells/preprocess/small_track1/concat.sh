op=$1

LANGS=(af am ar as ast ay az ba be bg bn br bs ca ceb cjk cs cy da de dyu el en es et fa ff fi fr fy ga gd gl gu ha he hi hr ht hu hy id ig ilo is it ja jv ka kac kam kea kg kk km kmb kmr kn ko ku ky lb lg ln lo lt luo lv mg mi mk ml mn mr ms mt my ne nl no ns ny oc om or pa pl ps pt qu ro ru sd shn si sk sl sn so sq sr ss su sv sw ta te tg th ti tl tn tr uk umb ur uz vi wo xh yi yo zh zu)
    
if [ "$op" == "small-task1-train" ]; then
    echo "Will Preprocess: small-task1..."
    #TEXT=/home/v-jiaya/SharedTask/data/large-scale/small_task1/download/small_task1_filt_spm/
    #TRAIN=/home/v-jiaya/SharedTask/data/large-scale/small_task1/download/train/
    TEXT=/mnt/input/SharedTask/large-scale/small_task1/download/small_task1_filt_spm/
    TRAIN=/mnt/input/SharedTask/large-scale/small_task1/download/train/
    mkdir -p $TRAIN
    for src in ${LANGS[@]}; do
        for tgt in ${LANGS[@]}; do
            echo "${src}->${tgt}"
            if [ ! -f $TRAIN/train.${src}-${tgt}.${src} -a ! -f $TRAIN/train.${src}-${tgt}.${tgt} ]; then
                count=$(ls $TEXT/*.${src}-${tgt}.${src} 2>/dev/null | wc -l)
                if [ $count -gt 0 ]; then
                    for file in $(ls $TEXT/*.${src}-${tgt}.${src}); do
                        file=$(basename $file)
                        echo "$TEXT/$file -> $TRAIN/train.${src}-${tgt}.${src}"            
                        cat $TEXT/$file >> $TRAIN/train.${src}-${tgt}.${src}
                    done
                    for file in $(ls $TEXT/*.${src}-${tgt}.${tgt}); do
                        file=$(basename $file)
                        echo "$TEXT/$file -> $TRAIN/train.${src}-${tgt}.${tgt}"            
                        cat $TEXT/$file >> $TRAIN/train.${src}-${tgt}.${tgt}
                    done      
                fi
            fi   
        done
    done
elif [ "$op" == "small-task2" ]; then
    echo "Will Preprocess: small-task2..."
    #TEXT=/home/v-jiaya/SharedTask/data/large-scale/small_task1/download/small_task1_filt_spm/
    #TRAIN=/home/v-jiaya/SharedTask/data/large-scale/small_task1/download/train/
    TEXT=/mnt/input/SharedTask/large-scale/small_task2/download/small_task2_filt_spm/
    TRAIN=/mnt/input/SharedTask/large-scale/small_task2/download/train/
    mkdir -p $TRAIN
    for src in ${LANGS[@]}; do
        for tgt in ${LANGS[@]}; do
            echo "${src}->${tgt}"
            count=$(ls $TEXT/*.${src}-${tgt}.${src} 2>/dev/null | wc -l)
            if [ $count -gt 0 ]; then
                for file in $(ls $TEXT/*.${src}-${tgt}.${src}); do
                    file=$(basename $file)
                    echo "$TEXT/$file -> $TRAIN/train.${src}-${tgt}.${src}"            
                    cat $TEXT/$file >> $TRAIN/train.${src}-${tgt}.${src}
                done
                for file in $(ls $TEXT/*.${src}-${tgt}.${tgt}); do
                    file=$(basename $file)
                    echo "$TEXT/$file -> $TRAIN/train.${src}-${tgt}.${tgt}"        
                    cat $TEXT/$file >> $TRAIN/train.${src}-${tgt}.${tgt}
                done
            fi          
        done
    done
else
    echo "Will Preprocess: large-task..."
    TEXT=/mnt/input/SharedTask/large-scale/small_task1/download/small_task1_filt_spm/
    TRAIN=/mnt/input/SharedTask/large-scale/small_task1/download/train/
    mkdir -p $TRAIN
    for src in ${LANGS[@]}; do
        for tgt in ${LANGS[@]}; do
            echo "${src}->${tgt}"
            count=$(ls $TEXT/*.${src}-${tgt}.${src} 2>/dev/null | wc -l)
            if [ $count -gt 0 ]; then
                for file in $(ls $TEXT/*.${src}-${tgt}.${src}); do
                    file=$(basename $file)
                    echo "$TEXT/$file -> $TRAIN/train.${src}-${tgt}.${src}"            
                    cat $TEXT/$file >> $TRAIN/train.${src}-${tgt}.${src}
                done
                for file in $(ls $TEXT/*.${src}-${tgt}.${tgt}); do
                    file=$(basename $file)
                    echo "$TEXT/$file -> $TRAIN/train.${src}-${tgt}.${tgt}"            
                    cat $TEXT/$file >> $TRAIN/train.${src}-${tgt}.${tgt}
                done      
            fi           
        done
    done
fi
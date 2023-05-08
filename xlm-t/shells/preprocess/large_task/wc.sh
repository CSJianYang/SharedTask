TEXT=$1
for dir in $(ls $TEXT); do
    wc -l $TEXT/$dir/*
done

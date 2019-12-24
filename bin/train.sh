# 训练数据
Date=$(date +%Y%m%d%H%M)
if [ "$1" = "" ]; then
    echo "train.sh help|stop|gpu"
    exit
fi

if [ "$1" = "help" ]; then
    echo "train.sh help|stop|gpu"
    exit
fi

if [ "$1" = "stop" ]; then
    echo "停止训练"
    ps aux|grep python|grep name=psenet|awk '{print $2}'|xargs kill -9
    exit
fi


echo "生产模式,使用GPU#$1"
nohup \
python -m train --gpu_list=$1 --input_size=512 --batch_size_per_gpu=8 --checkpoint_path=/app/models/pse/ctw1500 \
--training_data_path=./data/ctw1500/train/text_image \
--training_text_path=./data/ctw1500/train/text_label_curve/ \
>> ./logs/psenet_$1_$Date.log 2>&1 &
echo "启动完毕"
#--training_data_path=./data/ocr/icdar2015/
# 训练数据
Date=$(date +%Y%m%d%H%M)
if [ "$1" = "" ]; then
    echo "train.sh help|stop|console|gpu"
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

if [ "$1" = "console" ]; then
    echo "debug模式"
    python -m train \
    --name=psenet \
    --save_summary_steps=2 \
    --save_checkpoint_steps=2 \
    --max_steps=2 \
    --gpu_list=1 --input_size=512 \
    --batch_size_per_gpu=1 \
    --checkpoint_path=./model/ctw1500 \
    --training_data_path=./data/ctw1500/train/text_image \
    --training_text_path=./data/ctw1500/train/text_label_curve/
    exit
fi

echo "生产模式,使用GPU#$1"
nohup \
python -m train \
--name=psenet \
--gpu_list=$1 --input_size=512 --batch_size_per_gpu=1 \
--checkpoint_path=./model/ctw1500 \
--training_data_path=./data/ctw1500/train/text_image \
--training_text_path=./data/ctw1500/train/text_label_curve/ \
>> ./logs/psenet_$1_$Date.log 2>&1 &
echo "启动完毕"
#--training_data_path=./data/ocr/icdar2015/dar2015/
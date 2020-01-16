# 训练车牌检测数据
train_name=psenet_plate
data_type=plate


Date=$(date +%Y%m%d%H%M)
if [ "$1" = "" ]; then
    echo "train_plate.sh help|stop|console|gpu"
    exit
fi

if [ "$1" = "help" ]; then
    echo "train_icadr.sh help|stop|gpu"
    exit
fi

if [ "$1" = "stop" ]; then
    echo "停止训练"
    ps aux|grep python|grep name=$train_name|awk '{print $2}'|xargs kill -9
    exit
fi

if [ "$1" = "console" ]; then
    echo "debug模式 #$train_name #$data_type"

    python -m train \
    --name=$train_name \
    --data_type=$data_type \
    --save_summary_steps=2 \
    --save_checkpoint_steps=2 \
    --max_steps=2 \
    --num_readers=1 \
    --gpu_list=1 --input_size=512 \
    --batch_size_per_gpu=1 \
    --checkpoint_path=./model/plate \
    --training_data_path=./data/plate \
    --training_text_path=./data/plate/
    exit
fi

#TODO save_summary_steps 原来50
echo "生产模式,使用GPU#$1 #$train_name #$data_type"
nohup \
python -m train \
--name=$train_name \
--data_type=$data_type \
--save_summary_steps=10 \
--gpu_list=$1 --input_size=512 --batch_size_per_gpu=8 \
--num_readers=10 \
--checkpoint_path=./model/icdar2015 \
--training_data_path=./data/plate \
--training_text_path=./data/plate \
>> ./logs/psenet_plate_$1_$Date.log 2>&1 &
echo "启动完毕"
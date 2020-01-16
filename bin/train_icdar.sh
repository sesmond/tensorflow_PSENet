# 训练icdar数据
train_name=psenet_icdar2015
data_type=icdar


Date=$(date +%Y%m%d%H%M)
if [ "$1" = "" ]; then
    echo "train_icadr.sh help|stop|console|gpu"
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
    echo "debug模式"
    python -m train \
    --name=$train_name \
    --save_summary_steps=2 \
    --save_checkpoint_steps=2 \
    --max_steps=2 \
    --num_readers=1 \
    --gpu_list=1 --input_size=512 \
    --batch_size_per_gpu=1 \
    --checkpoint_path=./model/icdar2015 \
    --pretrained_model_path=./model/pred/model.ckpt \
    --training_data_path=./data/icdar2015/train/text_image \
    --training_text_path=./data/icdar2015/train/text_label_curve/
    exit
fi

echo "生产模式,使用GPU#$1"
nohup \
python -m train \
--name=$train_name \
--data_type=$data_type \
--save_summary_steps=50 \
--gpu_list=$1 --input_size=512 --batch_size_per_gpu=8 \
--num_readers=32 \
--checkpoint_path=./model/icdar2015 \
--training_data_path=./data/icdar2015/2015ch4_training_images/ \
--training_text_path=./data/icdar2015/2015ch4_training_localization_transcription_gt/ \
>> ./logs/psenet_icdar_$1_$Date.log 2>&1 &
echo "启动完毕"
#--pretrained_model_path=./model/pred/model.ckpt \
#--training_data_path=./data/ocr/icdar2015/dar2015/
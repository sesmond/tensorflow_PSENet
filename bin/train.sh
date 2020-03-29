# 训练数据
Date=$(date +%Y%m%d%H%M)


echo "选择您的操作"
select var in "train" "console" "stop"; do
  break;
done
echo "You have selected $var "


if [ "$var" = "stop" ]; then
    echo "停止训练"
    ps aux|grep python|grep name=psenet|awk '{print $2}'|xargs kill -9
    exit
fi


#echo "输入训练要用的GPU"
read -p "输入训练要用的GPU:" gpus ;

echo "您选择了GPU $gpus 进行训练"


if [ "$var" = "console" ]; then
    echo "debug模式"
    python -m train \
    --name=psenet \
    --save_summary_steps=2 \
    --save_checkpoint_steps=2 \
    --max_steps=2 \
    --num_readers=1 \
    --gpu_list=1 --input_size=512 \
    --batch_size_per_gpu=$gpus \
    --checkpoint_path=./model/ctw1500 \
    --pretrained_model_path=./model/pred/model.ckpt \
#    --training_data_path=./data/ctw1500/train/text_image \
#    --training_text_path=./data/ctw1500/train/text_label_curve/
    exit
fi

echo "生产模式,使用GPU#$gpus"
nohup \
python -m train \
--name=psenet \
--save_summary_steps=50 \
--gpu_list=$gpus \
--input_size=512 --batch_size_per_gpu=8 \
--num_readers=32 \
--checkpoint_path=model/multi \
--train_data_config=cfg/train_data.cfg \
>> ./logs/psenet_$Date.log 2>&1 &
echo "启动完毕,在logs下查看日志！"

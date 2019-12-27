if [ "$1" = "help" ]; then
    echo "pred.sh 使用说明：
    --evaluate      是否进行评价（你可以光预测，也可以一边预测一边评价）
    --split         是否对小框做出评价，和画到图像上
    --image_dir     被预测的图片目录
    --pred_dir     预测后的结果的输出目录
    --file          为了支持单独文件，如果为空，就预测test_home中的所有文件
    --draw          是否把gt和预测画到图片上保存下来，保存目录也是pred_dir
    --save          是否保存输出结果（大框、小框信息都要保存），保存到pred_dir目录里面去
    --ctpn_model    model的全路径 "
    exit
fi

echo "开始检测图片的字块区域....."

python  eval.py \
    --test_data_path=./data/pred/input/ \
    --checkpoint_path=./model/pred \
    --output_dir=./data/pred/output \
    --gpu_list=0 \

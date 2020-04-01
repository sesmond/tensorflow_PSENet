if [ "$1" = "help" ]; then
    echo "convert.sh 使用说明：
    --evaluate      是否进行评价（你可以光预测，也可以一边预测一边评价）"
    exit
fi

echo "开始转换模型....."

python  -m utils.convert_model\
    --ckpt_mod_path=model/multi/ \
    --save_mod_dir=model/multi_pb/

echo "转换结束！"
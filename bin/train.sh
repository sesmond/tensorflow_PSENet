# 训练数据
# 准备训练数据

python -m train --gpu_list=0 --input_size=512 --batch_size_per_gpu=8 --checkpoint_path=./model/ \
--training_data_path=/Users/minjianxu/Documents/icdar/ctw1500/train/text_image
#--training_data_path=./data/ocr/icdar2015/
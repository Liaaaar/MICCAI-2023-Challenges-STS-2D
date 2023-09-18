# MICCAI-2023-Challenges-STS-2D
Code for [MICCAI 2023 Challenges ：STS-基于2D 全景图像的牙齿分割任务](https://tianchi.aliyun.com/competition/entrance/532086/introduction?spm=a2c22.12281925.0.0.6c757137vqp2w7)
初赛成绩：0.9541  排名：37/1039      
复赛成绩：0.9583  排名：13/1039

代码说明：      
code/
1. data.py  dataset以数据预处理
2. data_unzip.py  解压原始数据
3. train.py  1 fold训练
4. infer.py  1 fold推理(with tta)
5. k_fold_train.py  k fold训练
6. k_fold_infer.py  k fold推理
7. model  所用到的模型(unet++ and deeplabv3+)
8. loss.py  所用到的损失函数(bce,dice,bce_dice)
9. remove_small_objects.py  数据后处理(移除面积小于阈值的区域)   

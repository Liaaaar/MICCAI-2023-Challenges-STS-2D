# MICCAI-2023-Challenges-STS-2D
Code for [MICCAI 2023 Challenges ：STS-基于2D 全景图像的牙齿分割任务](https://tianchi.aliyun.com/competition/entrance/532086/introduction?spm=a2c22.12281925.0.0.6c757137vqp2w7)       

## 成绩
- 初赛成绩：0.9541  排名：37/1039      
- 复赛成绩：0.9583  排名：13/1039

## 文件说明        
- code/    
  - data.py -- dataset以数据预处理 
  - data_unzip.py -- 解压原始数据    
  - train.py -- 1_fold训练    
  - infer.py -- 1_fold推理(with tta)    
  - k_fold_train.py -- k_fold训练    
  - k_fold_infer.py -- k_fold推理(with tta)    
  - model -- 所用到的模型(unet++ and deeplabv3+)    
  - loss.py -- 所用到的损失函数(bce,dice,bce_dice)    
  - remove_small_objects.py -- 数据后处理(移除面积小于阈值的区域)    
- infers.zip -- 复赛最好的结果

## 训练策略
### 初赛 
- 训练方式：全监督
- 架构：deeplabv3+, unet++, unet    
- backbone: resnet50, efficientnet_b5, se_resnext50_32x4d 
- 最终选用deeplabv3+和resnet50的方案(该方案最佳)
- tricks: 数据增强，tta(原图，竖直镜像，水平镜像，竖直和水平镜像)，k_fold_train，洪泛化(don't work)，数据后处理
### 复赛
- 训练方式：半监督(self_training)
- 所有设置与初赛保持一致，以初赛最优的单模型作为初始化，在复赛有标签数据上进行10_fold微调，每折保留最优的三个模型，分数为0.9580，数据后处理后分数为0.9583
- 使用分数为0.9580的模型为无标注数据生成成伪标签，之后将伪标签与有便签数据放在一起，重新训练一个10_fold模型，最终成绩为xxxx(最终结果没来得及提交)

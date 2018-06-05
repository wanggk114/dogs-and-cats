# Kaggle & Udacity 猫狗项目

# 文件说明
 ## 代码相关
| 执行顺序|  文件名 | 文件说明 | 机器 | 大概执行时间 |
| --- | --- | --- | --- | --- | 
| 1 | [pre_dir_process.ipynb](https://github.com/wanggk114/dogs-and-cats/blob/master/pre_dir_process.ipynb) |	建文件目录的代码 | p2 | 2min |
| 2 | [pick_up_anormal_images_batch_cat.ipynb](https://github.com/wanggk114/dogs-and-cats/blob/master/pick_up_anormal_images_batch_cat.ipynb) |	挑选训练集cat目录中异常图片的代码 | p2 | 15min |
| 3 | [pick_up_anormal_images_batch_dog.ipynb](https://github.com/wanggk114/dogs-and-cats/blob/master/pick_up_anormal_images_batch_dog.ipynb) |	挑选训练集dog目录中异常图片的代码 | p2 | 15min |
| 4 | [prep_data_process.ipynb](https://github.com/wanggk114/dogs-and-cats/blob/master/prep_data_process.ipynb) |	数据预处理的代码 | p2 | 1min |
| 5 | [visuals.ipynb](https://github.com/wanggk114/dogs-and-cats/blob/master/visuals.ipynb) |	探索性可视化的代码  | p2 | 5min |
| 6 | [keras_fine_tuning_Xception.ipynb](https://github.com/wanggk114/dogs-and-cats/blob/master/keras_fine_tuning_Xception.ipynb) |	Xception迁移学习的代码   | p2 | 130min |
| / | [keras_fine_tunig_InceptonResNetV2.ipynb](https://github.com/wanggk114/dogs-and-cats/blob/master/keras_fine_tunig_InceptonResNetV2.ipynb) |	InceptonResNetV2迁移学习 v1   | / | / |
| / | [keras_fine_tunig_InceptonResNetV2-memory.ipynb](https://github.com/wanggk114/dogs-and-cats/blob/master/keras_fine_tunig_InceptonResNetV2-memory.ipynb) |	InceptonResNetV2迁移学习 v2   | / | / |
| / | [keras_fine_tunig_InceptonResNetV2-memory-v3.ipynb](https://github.com/wanggk114/dogs-and-cats/blob/master/keras_fine_tunig_InceptonResNetV2-memory-v3.ipynb) |	InceptonResNetV2迁移学习 v3   | / | / |
| 7 | [keras_fine_tunig_InceptonResNetV2-memory-v4.ipynb](https://github.com/wanggk114/dogs-and-cats/blob/master/keras_fine_tunig_InceptonResNetV2-memory-v4.ipynb) |	InceptonResNetV2迁移学习 v4   | p3 | 90min |
| 8 | [keras_fine_tunig_InceptonResNetV2-memory-v5.ipynb](https://github.com/wanggk114/dogs-and-cats/blob/master/keras_fine_tunig_InceptonResNetV2-memory-v5.ipynb) |	InceptonResNetV2迁移学习 v5   | p3 | 90min |
| 9 | [keras_fine_tunig_ResNet50-v1.ipynb](https://github.com/wanggk114/dogs-and-cats/blob/master/keras_fine_tunig_ResNet50-v1.ipynb) |	ResNet50迁移学习的代码   | p3 | 40min |
| 10 | [keras_merge_3_app.ipynb](https://github.com/wanggk114/dogs-and-cats/blob/master/keras_merge_3_app.ipynb) |	融合模型，加载keras预训练权重   | p2 | 40min |
| 11 | [keras_merge_3_app_by_tuning.ipynb](https://github.com/wanggk114/dogs-and-cats/blob/master/keras_merge_3_app_by_tuning.ipynb) |	融合模型，加载迁移学习保存的预训练权重   | p2 | 40min |
| 12 | [keras_merge_3_app_predict.ipynb](https://github.com/wanggk114/dogs-and-cats/blob/master/keras_merge_3_app_predict.ipynb) |	鲁棒性验证代码   | p2 | 5min |
| / | [helper.py](https://github.com/wanggk114/dogs-and-cats/blob/master/helper.py) | 辅助函数  |  | / |
 

## 机器说明
p2,p3均为aws上的实例

## 预测结果	
  - pred-xception-freeze-2.csv	Xception放开输出层的预测结果<br>
  - pred-xception-fine-tuning-1.csv	Xception冻结前97层的预测结果<br>
  - pred-InceptonResNetV2-base-tuning-v4.csv	InceptonResNetV2 v4只训练输出层<br>
  - pred-InceptonResNetV2-fine-tuning-1-v4.csv	InceptonResNetV2 v4冻结前698层<br>
  - pred-InceptonResNetV2-fine-tuning-2-v4.csv	InceptonResNetV2 v4冻结前618层<br>
  - pred-InceptonResNetV2-fine-tuning-3-v4.csv	InceptonResNetV2 v4冻结前499层<br>
  - pred-InceptonResNetV2-base-tuning-v3.csv	InceptonResNetV2 v5只训练输出层<br>
  - pred-InceptonResNetV2-fine-tuning-1-v5.csv	InceptonResNetV2 v5冻结前698层<br>
  - pred-InceptonResNetV2-fine-tuning-2-v5.csv	InceptonResNetV2 v5冻结前618层<br>
  - pred-InceptonResNetV2-fine-tuning-3-v5.csv	InceptonResNetV2 v5冻结前746层<br>
  - pred-ResNet50-base-tuning-v1.csv	ResNet50只训练输出层<br>
  - pred-ResNet50-fine-tuning-1-v1.csv	ResNet50冻结前164层<br>
  - pred-ResNet50-fine-tuning-2-v1.csv	ResNet50冻结前142层<br>
  - pred-ResNet50-fine-tuning-3-v1.csv	ResNet50冻结前112层<br>
  - pred-Merge-tuning-v1.csv	ImageNet模型融合，算法：adadelta<br>
  - pred-Merge-tuning-2-v1.csv	ImageNet模型融合，算法：Adam<br>
  - pred-Merge-tuning-v2.csv	fine-tuning模型融合，算法：adadelta<br>
  - pred-Merge-tuning-2-v2.csv	fine-tuning模型融合，算法：Adam<br>   
 ## 模型图	
    model_Xception.png	迁移学习中，在Xception基础上搭建的模型图<br>            
    model_InceptionResNetV2.png	迁移学习中，在InceptionResNetV2基础上搭建的模型图<br>   
    model_ResNet50.png	迁移学习中，在ResNet50基础上搭建的模型图<br>    
    
 # 用到的主要库
 keras、sklearn、shutil、os、cv2、pandas、matplotlib、PIL、tqdm  
 

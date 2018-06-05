# dogs-and-cats
deep learning of Kaggle project
# 文件说明
 ## 代码文件
 - helper.py 	辅助函数<br>
 - visuals.ipynb	探索性可视化的代码<br>
 - pick_up_anormal_images_batch_dog.ipynb	挑选训练集dog目录中异常图片的代码 <br>
 - pick_up_anormal_images_batch_cat.ipynb	挑选训练集cat目录中异常图片的代码 <br>
 - prep_data_process.ipynb	                整理文件目录的代码 <br>
 - keras_fine_tuning_Xception.ipynb	Xception迁移学习的代码 <br>
 - keras_fine_tunig_InceptonResNetV2.ipynb	InceptonResNetV2迁移学习 v1 <br>
 - keras_fine_tunig_InceptonResNetV2-memory-v2.ipynb	InceptonResNetV2迁移学习 v2 <br>
 - keras_fine_tunig_InceptonResNetV2-memory-v3.ipynb	InceptonResNetV2迁移学习 v3 <br>
 - keras_fine_tunig_InceptonResNetV2-memory-v4.ipynb	InceptonResNetV2迁移学习 v4 <br>
 - keras_fine_tunig_InceptonResNetV2-memory-v5.ipynb	InceptonResNetV2迁移学习 v5 <br>
 - keras_fine_tunig_ResNet50-v1.ipynb	ResNet50迁移学习的代码 <br>
 - keras_merge_3_app.ipynb	融合模型，加载keras预训练权重 <br>
 - keras_merge_3_app_by_tuning.ipynb	融合模型，加载迁移学习保存的预训练权重 <br>
 - keras_merge_3_app_predict.ipynb	鲁棒性验证代码 <br>
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

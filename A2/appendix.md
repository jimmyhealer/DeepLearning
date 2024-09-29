---
title: "Lab2 flower 102 Classifiers Report Appendix"
author: "313832008 簡蔚驊"
CJKmainfont: "Noto Sans CJK TC"
fontsize: 12pt
geometry: "a4paper, top=0.6cm, bottom=1.4cm, left=0.8cm, right=0.8cm"
numbersections: true
toc: true
toc-title: "目錄"
---

# 附錄

## 模型比較

![](./assets/SimpleNet_120_64_False_CrossEntropyLoss_SGD_0.001_0_0_None_20240929002128.png)
![](./assets/ResNet18-ReLU_125_16_True_CrossEntropyLoss_SGD_0.001_0.9_0.0005_ReduceLROnPlateau_min_0.1_7_20240927040043.png)
![](./assets/ResNet18-pretrained_120_16_False_CrossEntropyLoss_SGD_0.001_0.9_0.0005_ReduceLROnPlateau_min_0.1_7_20240929024842.png)
![](./assets/ResNet34-LeakyReLU_250_32_True_CrossEntropyLoss_NAdam_0.001_0_0.0001_ReduceLROnPlateau_min_0.1_10_20240926204953.png)
![](./assets/ResNet34-pretrained_120_32_False_CrossEntropyLoss_NAdam_0.001_0_0.0001_ReduceLROnPlateau_min_0.1_7_20240928212401.png)
![](./assets/ResNet50-SiLU_120_32_True_CrossEntropyLoss_NAdam_0.001_0_0.0001_ReduceLROnPlateau_min_0.1_7_20240928154105.png)
![](./assets/ResNet50-pretrained_120_32_False_CrossEntropyLoss_NAdam_0.001_0_0.0001_ReduceLROnPlateau_min_0.1_6_20240929034444.png)
![](./assets/ResNet101-LeakyReLU_150_32_True_CrossEntropyLoss_NAdam_0.001_0_0.0001_ReduceLROnPlateau_min_0.1_10_20240926123023.png)
![](./assets/ResNet101-pretrained_120_32_False_CrossEntropyLoss_NAdam_0.001_0_0.0001_ReduceLROnPlateau_min_0.1_5_20240929122214.png)
![](./assets/ResNet152-LeakyReLU_150_32_True_CrossEntropyLoss_NAdam_0.001_0_0.0001_ReduceLROnPlateau_min_0.1_7_20240926063847.png)
![](./assets/ResNet152-pretrained_120_64_False_CrossEntropyLoss_NAdam_0.001_0_0.0001_ReduceLROnPlateau_min_0.1_5_20240929124104.png)

## Scheduler Comparison

![](./assets/ResNet34-ReLU_120_32_True_CrossEntropyLoss_Adam_0.001_0_0_None_20240929015802.png)
![](./assets/ResNet34-ReLU_120_32_True_CrossEntropyLoss_Adam_0.001_0_0_ReduceLROnPlateau_min_0.1_7_20240928175820.png)
![](./assets/ResNet34-ReLU_100_True_CrossEntropyLoss_Adam_0.001_0_0_CosineAnnealingLR_20_20240924122936.png)
![](./assets/ResNet34-ReLU_100_True_CrossEntropyLoss_Adam_0.001_0_0_StepLR_15_0.3_20240923140127.png)
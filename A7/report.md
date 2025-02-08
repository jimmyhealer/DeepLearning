---
title: "DL Lab7: Anomaly Detec1on"
author: "313832008 簡蔚驊"
CJKmainfont: "Noto Sans CJK TC"
fontsize: 12pt
geometry: "a4paper, top=0.6cm, bottom=1.4cm, left=0.8cm, right=0.8cm"
numbersections: true
---

# 摘要

在本次實驗中，實作了一個變分自編碼器 (VAE) 來進行異常檢測，主要是針對不良產品進行分類。比較了不同模型結構和參數設置的影響，並且嘗試了多種方式來提升模型的準確率。實驗結果顯示，不同編碼器的選擇、超參數的調整與訓練方式，對模型的表現有顯著影響。

# 背景

變分自編碼器是一種生成模型，其可以有效地從輸入數據中學習到隱含特徵，並通過重構誤差來辨別異常數據。在本次實驗中，使用五個資料集來對產品的好壞進行分類。

# 模型架構

VAE 主要由編碼器與解碼器兩部分組成。在編碼器中，輸入的 RGB 圖像會經過一系列卷積層，逐步減少空間維度，最終生成潛在變數的均值 (mu) 和對數方差 (logvar)。透過 reparameterization trick，從這些潛在變數中取樣，並將其輸入解碼器，解碼器則通過轉置卷積層將其還原為原始圖像。

## Encoder

### Convolutional Neural Network

總共包含五個卷積層，每層的輸出通道數分別為 32, 64, 128, 256, 512。每層的 kernel size 為 4，stride 為 2，padding 為 1。並使用 LeakyReLU 作為 activation function，及 Batch Normalization 來穩定訓練過程。

最後有兩層連結層，分別生成 mu 和 logvar。

```python
self.encoder = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
)
```

### ResNet18

使用預訓練的 ResNet18 作為編碼器，並移除最後的全連接層和 AvgPool，再加上一個 AdaptiveAvgPool2d 和 Flatten 層，最後再加上一個全連接層，輸出 mu 和 logvar。

```python
resnet18 = models.resnet18(pretrained=True)
self.encoder = nn.Sequential(
    *list(resnet18.children())[:-2],  # 移除最後的全連接層和 AvgPool
    nn.AdaptiveAvgPool2d((1, 1)),     # 將特徵圖縮小為 (1, 1)
    nn.Flatten(),
    nn.Linear(512, latent_dim * 2)    # 輸出 mu 和 logvar
)
```

### VIT-B-16

使用預訓練的 Vision Transformer (ViT) 作為編碼器，並移除最後的全連接層和 AvgPool，再加上一個 AdaptiveAvgPool2d 和 Flatten 層，最後再加上一個全連接層，輸出 mu 和 logvar。

```python
vit_b_16 = models.vit_b_16(weights='IMAGENET1K_V1')
self.encoder = nn.Sequential(
    *list(vit_b_16.children())[:-2],  # 移除最後的全連接層和 AvgPool
    nn.AdaptiveAvgPool2d((1, 1)),     # 將特徵圖縮小為 (1, 1)
    nn.Flatten(),
    nn.Linear(768, latent_dim * 2)    # 輸出 mu 和 logvar
)
```

## Decoder

解碼器與編碼器相反，總共包含五個轉置卷積層，每層的輸出通道數分別為 256, 128, 64, 32, 3。每層的 kernel size 為 4，stride 為 2，padding 為 1。最後一層使用 Tanh 作為 activation function，並將輸出範圍限制在 [-1, 1]。

```python
  self.decoder = nn.Sequential(
      nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
      nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
      nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
      nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
  )
```

# 實驗方法

## 多個資料集合併訓練

實驗兩種不同的訓練方法，一種是將五個資料集合併訓練，另一種是分別訓練每個資料集。

## 不同的 Encoder

嘗試了三種不同的 Encoder，分別是 Convolutional Neural Network、ResNet18 和 Vision Transformer (ViT)。

## 初始化

使用 Kaiming Normal 初始化卷積層的權重，Xavier Normal 初始化全連接層的權重，並將 Batch Normalization 的權重初始化為 1，偏差初始化為 0。

```python
def initialize_weights(self, m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
```

## Learning Rate 與 Scheduler

使用 AdamW 作為 Optimizer，並使用 CosineAnnealingLR 作為 Scheduler。並嘗試三種不同的 Learning Rate，分別是 1e-3、5e-4 和 1e-4。

<!-- ## 三分搜找最佳 Threshold

為了找到最佳的 Threshold，使用三分搜找到最佳的 Threshold，使得 F1 Score 最大。

```python
def find_best_threshold(low=0, high=2000, tolerance=1):
    while high - low > tolerance:
        mid1 = low + (high - low) // 3
        mid2 = high - (high - low) // 3

        # 計算兩個中間值的準確率和 F1 值
        accuracy1, f1_1 = calculate_accuracy_and_f1(mid1)
        accuracy2, f1_2 = calculate_accuracy_and_f1(mid2)

        # 根據準確率與 F1 值調整區間
        if accuracy1 > accuracy2:
            best_threshold, best_accuracy = mid1, accuracy1
            high = mid2 - 1
        elif accuracy1 == accuracy2:
            best_threshold, best_accuracy = mid1, accuracy1
            if f1_1 > f1_2:
                high = mid2 - 1
            else:
                low = mid1 + 1
        else:
            best_threshold, best_accuracy = mid2, accuracy2
            low = mid1 + 1

    return best_threshold, best_accuracy
``` -->

## 訓練超參數

以下訓練超參數為最終訓練結果：

- Batch Size: 64
- Epochs: 500
- Learning Rate: 5e-4
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- T_max: 100
- Latent Dim: 512

# 結果分析與比較

## 各個資料集的準確度

合併訓練的平均準確率為 85.34%，加權平均準確率為 82.92%。單獨訓練的平均準確率為 84.21%，加權平均準確率為 83.27%。

兩種訓練方式效果其實相當接近，若準確用加權平均準確率來看，單獨訓練的效果略好於合併訓練。

![各資料集準確率](./A7/assets/train_result.png){ width=100% }

由於測試資料集大多數為錯誤產品，如果將所有資料預測為錯誤，準確率仍可維持在較高水準。為了評估模型的實際能力，計算了模型對於「全預測錯誤準確率」的提升幅度，各資料集的改善情況如下：

![各資料集提升準確率](./A7/assets/improvements.png){ width=100% }

合併訓練讓牙刷這個資料集的提升幅度最大，一方面可能牙刷訓練資料較少，另一方面可能牙刷的特徵較為獨特，因此合併訓練能夠提升模型對於牙刷的辨識能力。

從圖中可以看出，事實上 pill 和 capsule 無論合併或單獨訓練幾乎都沒有提升，算是比較失敗的部分。

![合併訓練的 Loss 曲線](./A7/assets/loss_vae_all.png)

從合併訓練相比單獨訓練的 Loss 曲線後期容易 Overfitting。

## 不同 Encoder 的比較

以下使用 bottle 這個資料集來比較不同 Encoder 的效果。一般 CNN 模型的效果最好，ResNet18 次之，ViT 最差。可能是因為資料集太小，ResNet18 和 ViT 都是使用預訓練模型，因此效果不如自己訓練的 CNN 模型。

- Convolutional Neural Network: 90.36%
- ResNet18: 83.13%
- Vision Transformer (ViT): 79.51%

## Learning Rate 的比較

以下使用 Convolutional Neural Network 這個 Encoder 來比較不同 Learning Rate 的效果。Learning Rate 為 5e-4 時效果最好，1e-3 次之，1e-4 最差。

- 1e-3: 80.13%
- 5e-4: 90.36%
- 1e-4: 85.34%

# 總結

本次實驗成功實作了基於變分自編碼器 (VAE) 的異常檢測系統，並對不同模型架構、超參數設置與訓練策略進行了深入的比較與分析，主要結論如下：

- 合併與單獨訓練的比較
   - 合併訓練能夠提升模型對少數資料集（如牙刷）的辨識能力，但在部分資料集（如 pill 和 capsule）中表現一般。
   - 單獨訓練略微提高了加權平均準確率，尤其對於訓練數據特徵較為集中的資料集效果較佳。
- 不同 Encoder 的影響
   - CNN 表現最佳，其簡單且高效的特徵提取能力適合小型資料集。
   - ResNet18 雖使用預訓練權重，但因模型複雜度高，對小型資料集的適應性較差。
   - ViT 對大規模資料集表現優異，但在本次實驗中因資料量有限，學習效果不如預期。
- 超參數調整的重要性  
   - 不同的 Learning Rate 顯著影響模型表現，實驗中 Learning Rate 為 5e-4 時效果最佳，展示了適當的學習速率對於模型收斂與準確率提升的重要性。
   - 使用 Kaiming Normal 和 Xavier Normal 進行權重初始化，穩定了模型的訓練過程，避免了梯度爆炸或消失的問題。
- Loss 曲線與 Overfitting  
   - 合併訓練的 Loss 曲線在後期呈現 Overfitting 的跡象，說明在合併訓練下，模型對某些資料集過度擬合。需要透過 L2 正則化技術或資料增強來進一步改善。
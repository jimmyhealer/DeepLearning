---
title: "DL Lab6: Image Generative Models"
author: "313832008 簡蔚驊"
CJKmainfont: "Noto Sans CJK TC"
fontsize: 12pt
geometry: "a4paper, top=0.6cm, bottom=1.4cm, left=0.8cm, right=0.8cm"
numbersections: true
header-includes:
  - \usepackage{multirow}
---

# 摘要

本次實驗的目標是利用生成對抗網絡（GAN）生成花卉圖片，並對比不同模型架構對於生成效果的影響。具體來說，實作了 WGAN, WGAN-GP 和助教提供的 DCGAN，並使用 FLOWER-102 資料集進行訓練，針對不同的模型參數和結構進行比較與分析。

# 背景

生成對抗網絡（GAN）是深度學習中一種常用於生成圖片的技術。GAN 由生成器（Generator）和判別器（Discriminator）組成，雙方互相對抗，最終使得生成器可以生成類似於真實圖片的資料。本次實驗使用 FLOWER-102 資料集來進行訓練，目標是生成逼真的花卉圖像，並對 WGAN, WGAN-GP 與 DCGAN 的表現進行比較。

# 模型使用

在本次實驗中，我們實作 WGAN, WGAN-GP 並和 DCGAN 比較。

- **DCGAN**：是基於卷積神經網絡（CNN）的生成對抗網絡。生成器使用了一系列的轉置卷積層來生成圖像，判別器則使用卷積層來對輸入圖像進行分類。
- **WGAN**：相比於原始的 GAN，WGAN 使用 Wasserstein 距離來評估生成圖片與真實圖片的差異，並去除了 sigmoid 激活函數，以便損失函數更平滑。WGAN 引入了權重裁剪以滿足 Lipschitz 條件，這使得生成器的學習更加穩定。
- **WGAN-GP**：WGAN 的一個優化版，使用梯度懲罰（Gradient Penalty）來代替權重裁剪，以更好地實現 Lipschitz 連續性。

## Wasserstein Distance

Wasserstein 距離是一種用於衡量兩個概率分布之間的距離的方法。在 GAN 中，Wasserstein 距離被用來替代原始的 JS 散度，以更好地評估生成器生成的圖片與真實圖片之間的差異。

$$ W(P_r, P_g) = \inf_{\gamma \in \Pi(P_r, P_g)} \mathbb{E}_{(x, y) \sim \gamma} [||x - y||] $$

這裡，$P_r$ 是真實資料的分布，$P_g$ 是生成資料的分布，$\gamma$ 表示所有可能的聯合分布，將 $P_r$ 和 $P_g$ 配對，描述將真實資料分布移動到生成資料分布所需的最小成本。

## N_Critic

N_Critic 是 WGAN 中的一個重要超參數，它表示每次更新生成器前，評論器需要更新的次數。這個參數的設置對於 WGAN 模型的穩定訓練至關重要，因為評論器需要比生成器更充分地學習資料分布。通常將 N_Critic 設置為 5 或更高值，能夠幫助評論器學到更準確的 Wasserstein 距離，從而促進更穩定的生成器訓練。

## Weight Clipping and Gradient Penalty

WGAN 使用權重裁剪（weight clipping）來滿足 Lipschitz 連續性的約束。Clip Value 是一個用來限制評論器權重範圍的參數，所有評論器的權重都會被裁剪到 [-Clip Value, Clip Value] 之間，以確保評論器的梯度不會爆炸。然而，Clip Value 過大可能會導致辨別器無法學習到足夠的信息。

而為了解決這個問題，WGAN-GP 引入了梯度懲罰（Gradient Penalty），通過對評論器的梯度進行懲罰，使得評論器的梯度保持在一個合理的範圍內。

$$ \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}[(||\nabla_{\hat{x}} D(\hat{x})||_2 - 1)^2] $$

$\hat{x}$ 是在真實與生成圖片之間進行隨機插值得到的點，而 $\lambda$ 是控制懲罰強度的參數。

# 實驗方法

## 資料集準備

使用 FLOWER-102 資料集，將圖片轉換為 64x64，並進行標準化，並將 train, test 和 valid 合併成一個資料集，總共有 8189 張圖片。

## 訓練可重現性

在開發深度學習模型時，設置隨機種子有助於確保結果的可重現性，比如在 Jupyter Notebook 中使用 `torch.manual_seed(12)`。但由於 Notebook 的執行順序不固定，可能會隨意執行多個區塊，即使最初設置了隨機種子，也可能導致結果變異。

為了解決這個問題，通常的方法是在每個訓練區塊手動添加 `torch.manual_seed(12)`，這既繁瑣又容易遺漏。為了簡化此操作，可以利用 iPython 的事件系統，自動設置隨機種子：

```python
from IPython import get_ipython

def set_seed(seed=12):
    torch.manual_seed(seed)
    ...
    torch.backends.cudnn.deterministic = True

ip.events.register('pre_run_cell', set_seed)
```

然而，WGAN-GP 的梯度懲罰執行自動微分計算梯度的範數，這需要使用非確定性行為來最佳化，故 WGAN-GP 並無設定此隨機種子。此外，提交的作業仍沿用助教提供的隨機種子設置方法，未使用上述 iPython 事件系統。

## 初始化

在原始代碼中提供了使用正態分佈的初始化方法，現在新增一個 Xavier 初始化方法。

```python
def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```

## n_critic 訓練方法

在 WGAN 論文中指出，設置 `n_critic` 是為了讓評論器更精確地學習 Wasserstein 距離，從而為生成器提供穩定的學習方向。實作中有兩種常見方法：

1. **相同 batch 資料訓練 n_critic 次，再訓練生成器一次**：這樣的方式簡單，但可能導致評論器過度依賴單一批次數據，限制對整體分布的理解。

2. **不同 batch 資料訓練 n_critic 次，再訓練生成器一次**：每次使用不同的數據批次來更新評論器，讓其更全面地學習數據分布，通常能得到更好的效果。

### 實作細節

對於不同 batch 資料，常用的方法有：

- **`next(iter)` 取得 batch**：這會頻繁重新創建迭代器，導致性能瓶頸和資源浪費。
- **`cycle` 轉換為循環迭代器**：將 dataloader 包裝成無限循環迭代器，避免重複創建，提高效率：

  ```python
  from itertools import cycle
  dataloader_iter = cycle(dataloader)
  ```

使用 `cycle` 可以高效獲取下一個 batch，節省運算資源，避免 `next(iter)` 帶來的開銷。

## 訓練設定

| **Epochs** | **Batch Size** | **Learning Rate** | **Clip Value** | **N_Critic** | **Lambda** |
|------------|----------------|-------------------|----------------|--------------|------------|
| 150        | 64             | 0.0001            | 0.01           | 5            | 10         |

# 結果分析與比較

## WGAN 與 WGAN-GP 訓練效果比較

在這次實驗中，我們比較了 WGAN 和 WGAN-GP 的訓練效果。從 FID（Fréchet Inception Distance）分數來看，WGAN 的 FID 分數為 **54.22**，而 WGAN-GP 的 FID 分數為 **103.1**，顯示 WGAN 的生成效果明顯更好。

下圖 \ref{fig:wgan_and_wgan_gp} 是 WGAN Loss 和 WGAN-GP Loss 的比較，可以看到 WGAN-GP 的 Loss 曲線更加不穩定，而 WGAN 的 Loss 曲線較為平滑。從生成效果來看，WGAN 生成的花卉圖片更加逼真，細節更加豐富，如圖 \ref{fig:wgan_and_wgan_gp_result} 所示。

![WGAN Loss (左圖) 和 WGAN-GP Loss (右圖)](./A6/assets/wgan_and_wgan_gp.png){ #fig:wgan_and_wgan_gp }

![WGAN (左圖) 和 WGAN-GP (右圖) 生成的花卉圖片](./A6/assets/wgan_and_wgan_gp_result.png){ #fig:wgan_and_wgan_gp_result }

## 超參數調整實驗結果

在本次實驗中，我對 WGAN 模型進行了多次超參數調整，涵蓋了訓練輪數（EPOCH）、`n_critic` 設定、實作細節以及初始化方法。以下是整理的結果表格，並附上每組實驗的 FID 分數，以衡量生成效果的好壞。

- **訓練輪數 (EPOCH)**：增加 EPOCH 數可以顯著改善 FID 分數，顯示訓練時間對生成效果有正面影響。從 30 EPOCH 的 177.13 降到 150 EPOCH 的 54.22，說明更長的訓練時間能夠提升模型性能。
- **n_critic 設定**：將 `n_critic` 設為 5 一般效果較佳，特別是在長時間訓練時。與 `n_critic = 10` 相比，`n_critic = 5` 的 FID 分數更低，顯示更合適的更新頻率能幫助生成器學習得更好。
- **實作細節**：對於 `n_critic` 的訓練方式，使用不同 Batch 資料進行多次更新比相同 Batch 資料效果稍差，說明數據多樣性雖有一定幫助，但並非絕對優勢。
- **初始化方法**：Xavier 初始化效果較差，FID 分數為 85.28，而使用正態分佈初始化時，FID 分數為 58.63，顯示初始化方法對模型表現有顯著影響。

\begin{table}[h]
\centering
\begin{tabular}{|l|l|l|l|l|l|}
\hline
\textbf{維度} & \textbf{設置} & \textbf{EPOCH} & \textbf{n\_critic} & \textbf{實作細節} & \textbf{FID 分數} \\
\hline
\multirow{4}{*}{訓練輪數 (EPOCH)} &  & 30 & 5 & 相同 Batch 資料 & 177.13 \\
 &  & 80 & 10 & 相同 Batch 資料 & 83.07 \\
 &  & 120 & 5 & 相同 Batch 資料 & 58.63 \\
 &  & 150 & 5 & 相同 Batch 資料 & \textbf{54.22} \\
\hline
\multirow{4}{*}{n\_critic 設定} & n\_critic = 5 & 150 & 5 & 相同 Batch 資料 & \textbf{54.22} \\
 &  & 120 & 5 & 相同 Batch 資料 & 58.63 \\
 & n\_critic = 10 & 80 & 10 & 相同 Batch 資料 & 83.07 \\
 &  & 150 & 10 & 相同 Batch 資料 & 76.13 \\
\hline
\multirow{3}{*}{實作細節} & 相同 Batch 資料 & 150 & 5 & 相同 Batch 資料 & \textbf{54.22} \\
 & 不同 Batch 資料 & 150 & 5 & 不同 Batch 資料 & 60.39 \\
 &  & 120 & 5 & 不同 Batch 資料 & 60.18 \\
\hline
\multirow{2}{*}{Xavier 初始化} & Xavier 初始化 & 120 & 5 & 相同 Batch 資料 & 85.28 \\
 & 正態分佈初始化 & 120 & 5 & 相同 Batch 資料 & \textbf{58.63} \\
\hline
\end{tabular}
\end{table}

# 總結

本次實驗比較了不同生成對抗網絡（GAN）架構和超參數對生成花卉圖片的影響。我們實作了 WGAN 和 DCGAN，並使用 FLOWER-102 資料集進行訓練，進一步分析 WGAN 和 WGAN-GP 的性能差異。結果顯示，WGAN 的 FID 分數顯著優於 WGAN-GP，生成效果更好。

延長訓練輪數能夠明顯提升生成質量。例如，從 30 到 150 EPOCH，FID 分數從 177.13 降至 54.22，顯示訓練時間的影響很大。對於 `n_critic` 的設置，`n_critic = 5` 優於 `n_critic = 10`，長時間訓練下生成器更穩定。比較相同 Batch 資料與不同 Batch 資料的訓練效果，結果顯示使用相同 Batch 資料多次更新更具優勢。

初始化方法對模型影響也很顯著，正態分佈初始化的 FID 分數（58.63）優於 Xavier 初始化（85.28），說明適當的初始化方式對生成效果有積極作用。

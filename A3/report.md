---
title: "DL Lab 3: Plot Loss Curve"
author: "313832008 簡蔚驊"
CJKmainfont: "Noto Sans CJK TC"
fontsize: 12pt
geometry: "a4paper, top=0.6cm, bottom=1.4cm, left=0.8cm, right=0.8cm"
numbersections: true
---

# 簡介

這次作業的目標是從三個不同的 log 檔案中繪製 Loss 曲線，這些檔案分別是 `train_losses_log.log`、`test_losses_log.log` 和 `metrics_log.log`。這些檔案包含了多種評估指標，例如：`total_loss`, `cos`, `mse`, `bce`, `ssim`, `dice` 和 `kl`。

# 方法

## 資料讀取與處理

這三個 log 檔案包含了多個 Loss 指標，並使用 `re` 模組來解析這些數據。我使用正則表達式來提取出每個指標對應的數值。

1. **train_losses_log.log**: 檔案解析並提取出 `total_loss`, `cos`, `mse`, `bce`, `ssim`, `dice`, 和 `kl` 對應的欄位數據。
2. **test_losses_log.log**: 檔案解析並提取出 `total_loss`, `mse`, `bce`, `ssim`, 和 `dice` 對應的欄位數據。
3. **metrics_log.log**: 檔案解析並提取出 `IoU`, `mse` 和 `ssim` 對應的欄位數據。

## 繪製曲線

使用 `matplotlib` 庫來繪製三張不同的圖，分別對應每個 log 檔案。每張圖包含數條曲線，代表不同的指標。

### 圖像合併

在生成每個 log 檔案的單獨圖像後，我使用 `subplot` 將它們合併成一張圖片。

## 結論

這次作業提供了練習 `plt` 庫的機會。通過這個過程，我學會如何通過視覺化來分析模型的性能。`matplotlib` 庫的靈活性使我能夠根據具體需求進行自定義。

此外，這次作業也讓我增加了使用正則表達式（Regular Expression）的經驗。在處理 log 檔案時，正則表達式被用來高效地從中解析出不同的評估指標與數值，這提高了我對資料清洗與處理的掌握度。正則表達式的強大功能讓我能快速定位所需數據，進而減少了手動處理的時間，提高了工作效率。
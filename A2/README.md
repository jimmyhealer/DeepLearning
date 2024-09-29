## 紀錄

v1: 很舊的版本，架構有錯誤
v2: 修正架構
v3: 加入了一些新的功能
v4: 加入 val_dataloader
v5: 將之前的 random transform 合併到原有的 transform 中，變成兩倍資料量

### Model 錯誤

```python
    def make_layer(self, block, out_channels, blocks, stride=1, activation_function=nn.ReLU):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, activation_function))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            _b = block(self.in_channels, out_channels, 1, activation_function)
            layers.append(_b)
        return nn.Sequential(*layers)
```

```
_b = block(self.in_channels, out_channels, 1, activation_function)
```

這行原本是錯誤的，stride 應該要是 1 而不是 stride，導致多了 Downsample，而 resnet18 並不明顯有錯誤，直到 resnet34 才會出現錯誤

### ReduceLROnPlateau

這個應該要用 val_loss 處理，但現在壓根沒有 val_dataloader，所以要增加一個 val_dataloader
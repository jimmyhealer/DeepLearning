from PIL import Image

# 讀取圖片
image1 = Image.open('/home/jimmyhealer/Documents/DeepLearning/A6/assets/wgan_result.png')
image2 = Image.open('A6/assets/wgan_gp_result.png')

# 確保兩張圖片的高度相同，如果需要可以調整大小
image1 = image1.resize((image1.width, image2.height))
image2 = image2.resize((image2.width, image1.height))

# 創建合併後的圖片，寬度為兩張圖片寬度相加，高度取決於其中一張圖片的高度
merged_image = Image.new('RGB', (image1.width + image2.width, image1.height))

# 將圖片粘貼到新圖片中
merged_image.paste(image1, (0, 0))
merged_image.paste(image2, (image1.width, 0))

# 保存或顯示結果
merged_image.save('./A6/assets/wgan_and_wgan_gp_result.png')
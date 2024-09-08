
# Instagram Reels Renk Analizi: Makine Öğrenmesi ile Video Etkileşim Tahmini

Bu proje, Instagram Reels videolarının baskın renklerini analiz ederek, bu renklerin etkileşim oranları ile nasıl ilişkili olduğunu incelemeyi amaçlar. Videolardan alınan kareler üzerindeki baskın renkler, izlenme sayıları ve etkileşim oranlarıyla kıyaslanarak, hangi renk tonlarının daha fazla etkileşim getirdiği analiz edilir.

## Gerekli Veri Başlıkları:
- `video_id`: Videonun kimliği.
- `dominant_color`: Videodaki baskın renk (örneğin, RGB formatında).
- `views`: Videonun izlenme sayısı.
- `likes`: Videonun beğeni sayısı.
- `comments`: Yorum sayısı.
- `shares`: Paylaşım sayısı.
- `saves`: Kaydedilme sayısı.
- `engagement_rate`: Etkileşim oranı (likes + comments + shares + saves) / views.

## Proje Akışı:
1. **Videodan Kare Yakalama**: OpenCV kullanarak her videodan belirli aralıklarla kareler yakalayacağız.
2. **Baskın Renkleri Bulma**: K-means algoritması kullanarak videolardaki baskın renkleri bulacağız.
3. **Etkileşim Analizi**: Videoların izlenme sayıları ve etkileşim oranlarıyla baskın renkleri kıyaslayacağız.

## Gereksinimler:
Proje için gerekli Python kütüphaneleri:
```bash
pip install pandas opencv-python scikit-learn matplotlib
```

## Python Kodu:

```python
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Etkileşim oranını hesaplayan fonksiyon
def calculate_engagement_rate(df):
    df['engagement_rate'] = (df['likes'] + df['comments'] + df['shares'] + df['saves']) / df['views']
    return df

# Videodan baskın renkleri bulma
def get_dominant_color(image, k=4):
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)
    dominant_color = kmeans.cluster_centers_.astype(int)
    return dominant_color

# Videodan kare yakalama
def capture_frame_from_video(video_path, frame_time=2):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
    success, frame = video.read()
    if success:
        return frame
    else:
        return None

# Veri setini yükleyelim
df = pd.read_csv("video_data.csv")

# Videolardan baskın renkleri bulalım
dominant_colors = []
for video_id in df['video_id']:
    video_path = f'videos/{video_id}.mp4'  # Videoların olduğu klasörü belirtin
    frame = capture_frame_from_video(video_path, frame_time=2)
    if frame is not None:
        dominant_color = get_dominant_color(frame)
        dominant_colors.append(dominant_color)

df['dominant_color'] = dominant_colors

# Etkileşim oranını hesaplayalım
df = calculate_engagement_rate(df)

# Model oluşturma (Random Forest Regressor)
X = df[['dominant_color']]  # Renk özelliği
y = df['engagement_rate']  # Etkileşim oranı (hedef)

# Veriyi eğitim ve test olarak ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğitelim
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test verisiyle tahmin yapalım
y_pred = model.predict(X_test)

# Sonuçları ekrana yazdıralım
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

predictions = pd.DataFrame({
    'Actual Engagement Rate': y_test,
    'Predicted Engagement Rate': y_pred
})

print(predictions.head())

# Tahmin sonuçlarını görselleştirelim
plt.scatter(y_test, y_pred)
plt.xlabel('Gerçek Etkileşim Oranı')
plt.ylabel('Tahmin Edilen Etkileşim Oranı')
plt.title('Etkileşim Oranı Tahmin Sonuçları')
plt.show()
```

## Sonuçlar ve Yorumlar:
Bu proje, Instagram Reels videolarındaki baskın renklerin etkileşim oranı üzerindeki etkisini analiz etmek için tasarlanmıştır. Modelin tahmin ettiği etkileşim oranları ile gerçek oranları kıyaslayarak hangi renklerin daha fazla etkileşim aldığını belirleyebilirsiniz.


# Instagram Reels Etkileşim Optimizasyonu: Makine Öğrenmesi ile Video Özelliklerini Tahmin Etme

Bu proje, Instagram Reels videolarının izlenme ve etkileşim oranlarını artırmaya yönelik bir makine öğrenmesi modeli geliştirmeyi amaçlamaktadır. Video başlığı uzunluğu, açıklama uzunluğu, hashtag sayısı ve video süresi gibi özellikleri kullanarak etkileşim oranını optimize edebilecek tahminlerde bulunacağız.

## Proje Yapısı

### 1. Veri Başlıkları:
- `video_id`: Videonun kimliği.
- `title_length`: Video başlığının uzunluğu (karakter sayısı).
- `description_length`: Video açıklamasının uzunluğu (karakter sayısı).
- `hashtags_count`: Videoda kullanılan hashtag sayısı.
- `length`: Videonun süresi (saniye).
- `views`: Videonun izlenme sayısı (hedef değişken).
- `likes`: Videonun beğeni sayısı.
- `comments`: Yorum sayısı.
- `shares`: Paylaşım sayısı.
- `saves`: Kaydedilme sayısı.
- `engagement_rate`: Etkileşim oranı (likes + comments + shares + saves) / views.

### 2. Veri Hazırlığı:
Veri setinde başlık, açıklama ve videonun özellikleri kullanılarak bir makine öğrenmesi modeli eğitilecek. Amaç, gelecekteki videoların izlenme sayısı ve etkileşim oranını tahmin edebilmek.

### 3. Model Eğitimi:
- **Random Forest Regressor** modeli kullanılarak izlenme sayıları tahmin edilecektir.
- Eğitim verisi, modelin doğruluğunu artırmak için `train_test_split` yöntemi ile eğitim ve test setlerine ayrılacaktır.

### 4. Gereksinimler:
Proje için gerekli Python kütüphaneleri şunlardır:
```bash
pip install pandas scikit-learn
```

### 5. Python Kodu:

```python
# Gerekli kütüphaneleri yükleyelim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Veri setini yükleyelim (CSV ya da veri kaynağına göre değiştir)
# Örneğin, data.csv adlı bir dosyan varsa:
df = pd.read_csv("data.csv")

# Veri setine etkileşim oranını ekleyelim
df['engagement_rate'] = (df['likes'] + df['comments'] + df['shares'] + df['saves']) / df['views']

# Model için kullanılacak özellikler
X = df[['title_length', 'description_length', 'hashtags_count', 'length']]  # Girdi değişkenleri
y = df['views']  # Tahmin etmek istediğimiz hedef değişken (izlenme sayısı)

# Veriyi eğitim ve test olarak ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelimizi oluşturalım (Random Forest kullanıyoruz)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test veri seti ile tahmin yapalım
y_pred = model.predict(X_test)

# Modelin başarımını değerlendirelim (MSE - Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Videolar için tahmin sonuçları
predictions = pd.DataFrame({
    'Actual Views': y_test,
    'Predicted Views': y_pred
})

# Tahmin sonuçlarını inceleyelim
print(predictions.head())
```

### 6. Proje Kurulumu:

1. **Repo'yu klonlayın**:
```bash
git clone https://github.com/kullanici_adiniz/reponuz.git
```

2. **Gerekli kütüphaneleri yükleyin**:
```bash
pip install -r requirements.txt
```

3. **Veri dosyasını (`data.csv`) ana dizine ekleyin.**

4. **Projeyi çalıştırın**:
```bash
python main.py
```

### 7. Sonuçlar ve Yorumlar:
Bu proje, video uzunluğu, başlık ve açıklama uzunluğu gibi değişkenlerle izlenme sayısının tahmin edilebileceğini gösteriyor. Modelin başarımı `Mean Squared Error` metriği ile değerlendirildi. Proje ilerledikçe daha fazla özellik ve daha karmaşık modeller eklenebilir.

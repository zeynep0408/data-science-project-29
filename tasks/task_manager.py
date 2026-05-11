"""
DS-29 — California Housing: Çoklu Lineer Regresyon + Multicollinearity
Modül: ML-02 (Regresyon) • Part 1568-1570 + 1631

Senaryo: Bir emlak danışmanlık şirketinde junior data scientist'sin. California
ev fiyatlarını feature'lardan (gelir, oda sayısı, konum...) tahmin eden bir
model kurman gerekiyor. Bonus: katsayılarının güvenilir olup olmadığını da
sorgulayacaksın — multicollinearity tuzağına düşmeden.

Her fonksiyonun `pass` kısmını doldur. Testleri çalıştır:
  python watch.py    # otomatik
  pytest tests/      # manuel
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. California Housing veri setini yükle
def load_california_data():
    """
    sklearn.datasets.fetch_california_housing kullanarak veri yükle.

    Returns:
        dict: {
            'X': feature DataFrame (shape (20640, 8)),
            'y': target Series — MedHouseVal (100.000 dolar cinsi),
            'feature_names': list — 8 feature adı,
        }
    """
    pass


# 2. Veriyi train ve test olarak böl + scale et
def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    train_test_split + StandardScaler uygula.

    Adımlar:
    - train_test_split(X, y, test_size=test_size, random_state=random_state)
    - StandardScaler:
      - Train'de fit_transform
      - Test'te SADECE transform (data leakage olmasın)

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    pass


# 3. Lineer regresyon modeli eğit
def train_linear_model(X_train_scaled, y_train):
    """
    sklearn LinearRegression ile model eğit.

    Returns:
        fit edilmiş LinearRegression nesnesi
    """
    pass


# 4. Model performansını değerlendir
def evaluate_model(model, X_test_scaled, y_test):
    """
    Modelin test setindeki MAE, RMSE ve R²'sini hesapla.

    Returns:
        dict: {'mae': float, 'rmse': float, 'r2': float}
    """
    pass


# 5. Katsayıları yorumla
def analyze_coefficients(model, feature_names):
    """
    Ölçekli modelin katsayılarını feature isimleriyle eşle.
    Mutlak değere göre en güçlüden en zayıfa sırala.

    Returns:
        pandas.DataFrame: columns=['feature', 'coef', 'abs_coef'],
                          sorted by abs_coef descending
    """
    pass


# 6. Yüksek korelasyonlu feature çiftlerini bul
def find_correlated_pairs(X, threshold=0.7):
    """
    Feature matrisinde |korelasyon| > threshold olan çiftleri bul.

    Args:
        X: DataFrame
        threshold: float (0-1 arası), default 0.7

    Returns:
        list of tuple: [(feature_a, feature_b, korelasyon), ...]
                       (her çift sadece bir kez, korelasyon değeriyle)
    """
    pass


# 7. VIF (Variance Inflation Factor) hesapla
def calculate_vif(X):
    """
    Her feature için VIF'i manuel hesapla.

    Formül: VIF_i = 1 / (1 - R²_i)
    Burada R²_i = "feature i'yi diğer feature'lardan tahmin eden regresyonun R²'si"

    Args:
        X: DataFrame (feature matrix)

    Returns:
        pandas.DataFrame: columns=['feature', 'vif'],
                          sorted by vif descending
    """
    pass


# 8. Yeni feature üret — multicollinearity çözümü
def engineer_bedrms_per_room(X):
    """
    AveBedrms ve AveRooms korelasyonlu (0.84) → ikisini birleştir.
    Yeni feature: BedrmsPerRoom = AveBedrms / AveRooms
    Eski iki feature'ı sil.

    Args:
        X: DataFrame (orijinal feature matrix, AveBedrms ve AveRooms içermeli)

    Returns:
        pandas.DataFrame: yeni feature'la güncellenmiş veri
                          (AveBedrms ve AveRooms silinmiş, BedrmsPerRoom eklenmiş)
    """
    pass


# 9. Yeni ev için fiyat tahmini
def predict_house_price(model, scaler, features_dict, feature_names):
    """
    Yeni bir ev için fiyat tahmini yap.

    Args:
        model: fit edilmiş LinearRegression
        scaler: fit edilmiş StandardScaler
        features_dict: dict — feature adı: değer (örn. {'MedInc': 5.0, 'HouseAge': 25, ...})
        feature_names: list — sıralı feature isimleri (scaler'ın beklediği sıra)

    Returns:
        float — tahmin edilen fiyat (100.000 dolar cinsi)
    """
    pass


# 10. Tam pipeline'ı çalıştır
def run_full_pipeline():
    """
    Yukarıdaki tüm fonksiyonları kullanarak baştan sona pipeline çalıştır.

    Adımlar:
    1. Veriyi yükle
    2. Split + scale
    3. Lineer model eğit
    4. Test setinde MAE/RMSE/R² hesapla
    5. Katsayıları yorumla

    Returns:
        dict: {
            'metrics': dict — MAE/RMSE/R²,
            'coefficients': DataFrame,
            'test_size': int,
            'train_size': int,
        }
    """
    pass

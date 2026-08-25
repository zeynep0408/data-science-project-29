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

from pyexpat import model

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. California Housing veri setini yükle
def load_california_data():
    data = fetch_california_housing(as_frame=True)
    return {
        'X': data.data,
        'y': data.target,
        'feature_names': list(data.feature_names),
    }

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
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
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model
    """
    sklearn LinearRegression ile model eğit.

    Returns:
        fit edilmiş LinearRegression nesnesi
    """
    pass


# 4. Model performansını değerlendir
def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}
    """
    Modelin test setindeki MAE, RMSE ve R²'sini hesapla.

    Returns:
        dict: {'mae': float, 'rmse': float, 'r2': float}
    """
    pass


# 5. Katsayıları yorumla
def analyze_coefficients(model, feature_names):
    df = pd.DataFrame({
        'feature': feature_names,
        'coef': model.coef_,
        'abs_coef': np.abs(model.coef_),
    })
    df = df.sort_values('abs_coef', ascending=False).reset_index(drop=True)
    return df
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
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    corr = X.corr()
    features = list(X.columns)
    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            c = corr.iloc[i, j]
            if abs(c) > threshold:
                pairs.append((features[i], features[j], float(c)))
    return pairs
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
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    features = list(X.columns)
    vif_list = []
    for i in range(len(features)):
        feature_i = X.iloc[:, i]
        others = X.drop(columns=[features[i]])
        r2_i = LinearRegression().fit(others, feature_i).score(others, feature_i)
        vif_i = 1 / (1 - r2_i) if r2_i < 1 else float('inf')
        vif_list.append(vif_i)
    df = pd.DataFrame({'feature': features, 'vif': vif_list})
    df = df.sort_values('vif', ascending=False).reset_index(drop=True)
    return df

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
    X_new = X.copy()
    X_new['BedrmsPerRoom'] = X_new['AveBedrms'] / X_new['AveRooms']
    X_new = X_new.drop(columns=['AveBedrms', 'AveRooms'])
    return X_new
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
    values = np.array([[features_dict[name] for name in feature_names]])
    values_scaled = scaler.transform(values)
    prediction = model.predict(values_scaled)
    return float(prediction[0])
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
    data = load_california_data()
    X, y = data['X'], data['y']
    feature_names = data['feature_names']

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(X, y)
    model = train_linear_model(X_train_scaled, y_train)
    metrics = evaluate_model(model, X_test_scaled, y_test)
    coefs = analyze_coefficients(model, feature_names)

    return {
        'metrics': metrics,
        'coefficients': coefs,
        'test_size': len(y_test),
        'train_size': len(y_train),
    }
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

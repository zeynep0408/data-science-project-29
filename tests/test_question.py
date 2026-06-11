import pytest
import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    load_california_data,
    split_and_scale,
    train_linear_model,
    evaluate_model,
    analyze_coefficients,
    find_correlated_pairs,
    calculate_vif,
    engineer_bedrms_per_room,
    predict_house_price,
    run_full_pipeline,
)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ─── 1. load_california_data ─────────────────────────────────

def test_load_data_returns_dict():
    data = load_california_data()
    assert isinstance(data, dict)
    assert set(data.keys()) == {'X', 'y', 'feature_names'}


def test_load_data_shapes():
    data = load_california_data()
    assert data['X'].shape == (20640, 8)
    assert data['y'].shape == (20640,)
    assert len(data['feature_names']) == 8


def test_load_data_has_medinc_feature():
    data = load_california_data()
    assert 'MedInc' in data['feature_names']
    assert 'AveRooms' in data['feature_names']
    assert 'AveBedrms' in data['feature_names']


# ─── 2. split_and_scale ─────────────────────────────────────

def test_split_and_scale_shapes():
    data = load_california_data()
    X_tr, X_te, y_tr, y_te, scaler = split_and_scale(data['X'], data['y'])
    assert X_tr.shape[0] == 16512  # %80
    assert X_te.shape[0] == 4128   # %20
    assert X_tr.shape[1] == 8
    assert isinstance(scaler, StandardScaler)


def test_split_and_scale_scaling_correct():
    """Scaling sonrası train'in ortalaması ~0, std ~1 olmalı"""
    data = load_california_data()
    X_tr, _, _, _, _ = split_and_scale(data['X'], data['y'])
    means = X_tr.mean(axis=0)
    stds = X_tr.std(axis=0)
    assert np.all(np.abs(means) < 1e-10), f"Means not ~0: {means}"
    assert np.all(np.abs(stds - 1.0) < 1e-2), f"Stds not ~1: {stds}"


# ─── 3. train_linear_model ──────────────────────────────────

def test_train_linear_model_returns_fitted():
    data = load_california_data()
    X_tr, _, y_tr, _, _ = split_and_scale(data['X'], data['y'])
    model = train_linear_model(X_tr, y_tr)
    assert isinstance(model, LinearRegression)
    assert hasattr(model, 'coef_')
    assert len(model.coef_) == 8


# ─── 4. evaluate_model ──────────────────────────────────────

def test_evaluate_model_returns_dict():
    data = load_california_data()
    X_tr, X_te, y_tr, y_te, _ = split_and_scale(data['X'], data['y'])
    model = train_linear_model(X_tr, y_tr)
    metrics = evaluate_model(model, X_te, y_te)
    assert set(metrics.keys()) == {'mae', 'rmse', 'r2'}


def test_evaluate_model_r2_reasonable():
    """California lineer regresyon ile R² ~0.58 civarı çıkmalı"""
    data = load_california_data()
    X_tr, X_te, y_tr, y_te, _ = split_and_scale(data['X'], data['y'])
    model = train_linear_model(X_tr, y_tr)
    metrics = evaluate_model(model, X_te, y_te)
    assert 0.50 < metrics['r2'] < 0.70, f"R² beklenen aralıkta değil: {metrics['r2']}"
    assert metrics['rmse'] > metrics['mae'], "RMSE her zaman MAE'den büyük ya da eşit olmalı"


# ─── 5. analyze_coefficients ────────────────────────────────

def test_analyze_coefficients_returns_dataframe():
    data = load_california_data()
    X_tr, _, y_tr, _, _ = split_and_scale(data['X'], data['y'])
    model = train_linear_model(X_tr, y_tr)
    df = analyze_coefficients(model, data['feature_names'])
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {'feature', 'coef', 'abs_coef'}
    assert len(df) == 8


def test_analyze_coefficients_sorted():
    """En büyük mutlak katsayı ilk sırada olmalı (MedInc beklenir)"""
    data = load_california_data()
    X_tr, _, y_tr, _, _ = split_and_scale(data['X'], data['y'])
    model = train_linear_model(X_tr, y_tr)
    df = analyze_coefficients(model, data['feature_names'])
    abs_coefs = df['abs_coef'].values
    assert np.all(abs_coefs[:-1] >= abs_coefs[1:]), "abs_coef azalan sırada olmalı"


# ─── 6. find_correlated_pairs ───────────────────────────────

def test_find_correlated_pairs_format():
    data = load_california_data()
    pairs = find_correlated_pairs(data['X'], threshold=0.7)
    assert isinstance(pairs, list)
    for pair in pairs:
        assert len(pair) == 3
        assert abs(pair[2]) > 0.7


def test_find_correlated_pairs_california_known():
    """California'da AveRooms-AveBedrms ve Latitude-Longitude şüpheli olmalı"""
    data = load_california_data()
    pairs = find_correlated_pairs(data['X'], threshold=0.7)
    pair_names = [(a, b) for a, b, c in pairs]
    pair_names_flat = [f"{a}-{b}" for a, b in pair_names] + [f"{b}-{a}" for a, b in pair_names]
    # AveRooms-AveBedrms ya da Latitude-Longitude'dan en az biri çıkmalı
    found = any('AveRooms-AveBedrms' in p or 'AveBedrms-AveRooms' in p
                or 'Latitude-Longitude' in p or 'Longitude-Latitude' in p
                for p in pair_names_flat)
    assert found, f"AveRooms-AveBedrms veya Latitude-Longitude beklenmiyor — bulunan: {pairs}"


# ─── 7. calculate_vif ───────────────────────────────────────

def test_calculate_vif_format():
    data = load_california_data()
    vif_df = calculate_vif(data['X'])
    assert isinstance(vif_df, pd.DataFrame)
    assert set(vif_df.columns) == {'feature', 'vif'}
    assert len(vif_df) == 8


def test_calculate_vif_sorted():
    """VIF azalan sırada gelmeli"""
    data = load_california_data()
    vif_df = calculate_vif(data['X'])
    vifs = vif_df['vif'].values
    assert np.all(vifs[:-1] >= vifs[1:])


def test_calculate_vif_high_for_correlated():
    """California'da Latitude/Longitude/AveRooms/AveBedrms VIF'i yüksek olmalı"""
    data = load_california_data()
    vif_df = calculate_vif(data['X'])
    high_vif = vif_df[vif_df['vif'] > 5]['feature'].tolist()
    expected_some = {'Latitude', 'Longitude', 'AveRooms', 'AveBedrms'}
    found = expected_some & set(high_vif)
    assert len(found) >= 2, f"En az 2 yüksek VIF feature beklenir, bulunan: {high_vif}"


# ─── 8. engineer_bedrms_per_room ────────────────────────────

def test_engineer_feature_columns():
    data = load_california_data()
    X_new = engineer_bedrms_per_room(data['X'])
    assert 'BedrmsPerRoom' in X_new.columns
    assert 'AveBedrms' not in X_new.columns
    assert 'AveRooms' not in X_new.columns
    assert X_new.shape[1] == 7  # 8 - 2 + 1


def test_engineer_feature_values_reasonable():
    """BedrmsPerRoom oran olduğu için 0-1 arası olmalı"""
    data = load_california_data()
    X_new = engineer_bedrms_per_room(data['X'])
    assert X_new['BedrmsPerRoom'].mean() > 0
    assert X_new['BedrmsPerRoom'].mean() < 1


# ─── 9. predict_house_price ─────────────────────────────────

def test_predict_house_price_returns_float():
    data = load_california_data()
    X_tr, _, y_tr, _, scaler = split_and_scale(data['X'], data['y'])
    model = train_linear_model(X_tr, y_tr)

    # Örnek bir ev — California ortalaması civarı
    sample = {
        'MedInc': 5.0,
        'HouseAge': 25.0,
        'AveRooms': 5.5,
        'AveBedrms': 1.0,
        'Population': 1500,
        'AveOccup': 3.0,
        'Latitude': 34.0,
        'Longitude': -118.0,
    }
    pred = predict_house_price(model, scaler, sample, data['feature_names'])
    assert isinstance(pred, float)
    assert 0 < pred < 10, f"Tahmin makul aralıkta değil: {pred}"


# ─── 10. run_full_pipeline ──────────────────────────────────

def test_run_full_pipeline_returns_dict():
    result = run_full_pipeline()
    assert set(result.keys()) == {'metrics', 'coefficients', 'test_size', 'train_size'}
    assert result['test_size'] + result['train_size'] == 20640
    assert 0.50 < result['metrics']['r2'] < 0.70


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()

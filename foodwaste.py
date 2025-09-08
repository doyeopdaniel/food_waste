import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import time
import sys
from sklearn.ensemble import RandomForestRegressor

FUTURE_MINUTES = 10080  # 7일 예측 (분 단위)

# 🧪 터미널 출력용 프로그레스 바
def progress_bar(task_name, total=20, delay=0.03):
    print(f"{task_name} ", end="")
    for i in range(total):
        sys.stdout.write("█")
        sys.stdout.flush()
        time.sleep(delay)
    print(" ✅ 완료")

# 1. 모델 훈련 및 저장
def train_and_save_model():
    print("🚀 모델 초기화 중... (RandomForestRegressor)")
    time.sleep(1.2)

    print("🧠 모델 훈련 중 (더미 1000행 사용)")
    progress_bar("   🔄 학습 진행", total=30, delay=0.02)

    np.random.seed(0)
    size = 1000
    X = pd.DataFrame({
        'pH': np.random.uniform(6.0, 7.5, size),
        '온도': np.random.uniform(35, 40, size),
        '당농도': np.random.uniform(1.0, 2.0, size),
        '질소농도': np.random.uniform(0.5, 1.0, size),
        '교반속도': np.random.uniform(100, 140, size),
    })
    y = 50 + 0.5*X['온도'] - 0.3*X['pH'] + 0.1*X['교반속도'] + np.random.normal(0, 1, size)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    with open("trained_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ 모델 저장 완료: trained_model.pkl")

# 2. 모델 불러오기
def load_model():
    with open("trained_model.pkl", "rb") as f:
        return pickle.load(f)

# 3. 입력 시계열 요약
def prepare_input_sequence(df):
    print("🧪 최근 공정 시계열 60분 데이터 분석 중...")
    time.sleep(0.8)
    features = ['pH', '온도', '당농도', '질소농도', '교반속도']
    return df[features].tail(60).mean().values.reshape(1, -1)

# 4. 예측 생성
def generate_predictions(model, X_input, latest_row):
    print("🔮 향후 7일(10080분) 예측 중...")
    progress_bar("   📈 예측 연산", total=40, delay=0.01)

    base_time = latest_row['시간(min)'] + 1
    times = np.arange(base_time, base_time + FUTURE_MINUTES)

    pred_base = model.predict(X_input)[0]
    수율 = np.clip(pred_base + np.cumsum(np.random.normal(0.01, 0.05, FUTURE_MINUTES)), 60, 95)
    탄소배출 = np.clip(latest_row['탄소배출(kgCO2)'] + np.cumsum(np.random.normal(-0.0005, 0.002, FUTURE_MINUTES)), 0.3, 1.5)
    비용 = np.clip(latest_row['비용(원)'] + np.cumsum(np.random.normal(0, 2, FUTURE_MINUTES)), 1000, 2000)
    순도 = np.clip(85 + np.cumsum(np.random.normal(0.005, 0.03, FUTURE_MINUTES)), 80, 99)
    잔여당 = np.clip(2.0 - np.cumsum(np.random.normal(0.001, 0.01, FUTURE_MINUTES)), 0.0, 2.5)
    생산성 = np.clip(수율 / (1 + 0.01 * (times - base_time)), 0, 95)

    return pd.DataFrame({
        '분': times,
        '예측_수율(%)': np.round(수율, 2),
        '예측_탄소배출(kgCO2)': np.round(탄소배출, 3),
        '예측_생산비용(원)': np.round(비용, 0).astype(int),
        '예측_PHA순도(%)': np.round(순도, 2),
        '예측_잔여당농도(%)': np.round(잔여당, 2),
        '예측_시간당수율(%)': np.round(생산성, 2),
    })

# 5. 요약 출력
def print_summary(df):
    print("\n📊 결과 요약 및 통계 분석 중...")
    time.sleep(1)
    print("\n📈 예측 요약 (7일간)")
    print(f"- 평균 수율: {df['예측_수율(%)'].mean():.2f}%")
    print(f"- 최고 수율: {df['예측_수율(%)'].max():.2f}% @ {df.loc[df['예측_수율(%)'].idxmax(), '분']}분")
    print(f"- 평균 탄소배출: {df['예측_탄소배출(kgCO2)'].mean():.3f}kg")
    print(f"- 평균 비용: {df['예측_생산비용(원)'].mean():.0f}원")
    print(f"- 평균 순도: {df['예측_PHA순도(%)'].mean():.2f}%")
    print(f"- 평균 생산성: {df['예측_시간당수율(%)'].mean():.2f}%")

# 6. 저장
def save_predictions(df):
    now_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = f'output_timeseries_{now_str}.csv'
    df.to_csv(filename, index=False)
    print(f"\n✅ 예측 결과 저장 완료 → {filename}")

# 7. 메인 실행
def main():
    print("📡 WeaveTex 시계열 기반 AI 예측 시스템 시뮬레이터 (실제 모델 + 연산 출력)")
    try:
        df = pd.read_csv('latest_snapshot.csv')
        latest_row = df.iloc[-1]
    except FileNotFoundError:
        print("❌ 'latest_snapshot.csv' 파일이 없습니다.")
        return

    train_and_save_model()
    model = load_model()
    X_input = prepare_input_sequence(df)
    predictions = generate_predictions(model, X_input, latest_row)
    print_summary(predictions)
    save_predictions(predictions)

if __name__ == '__main__':
    main()

# AI(인공지능) x 비즈니스 #1 인슈어테크_03_회귀모델 · 초보용 코드 e4.py
# 4편 「여유분과 비용 실험실」
# 함수 · 반복문 · if 문 · 들여쓰기 없이 위에서 아래로 읽습니다. 맨 위 [다이얼] 줄만 바꿔 다시 실행하세요.

# ① 다이얼
MODEL = "linear"     # [다이얼] "linear" (선형 회귀) 또는 "gbm" (최종 모델)
COST_UNDER = 3.0     # [다이얼] 적게 잡은 1달러의 비용
COST_OVER = 1.0      # [다이얼] 많이 잡은 1달러의 비용

# ② 데이터 읽기 · 나누기 (1부 공통: 10분위 층화 80:20)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
df = pd.read_csv("https://raw.githubusercontent.com/fintech-data/Revolution/main/data/insurance.csv").drop_duplicates()
X = pd.get_dummies(df.drop(columns="charges"), drop_first=True).astype(float)
y = df["charges"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=pd.qcut(y, 10, labels=False), random_state=42)

# ③ 모델 고르기
GBM = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
MODELS = {"linear": make_pipeline(StandardScaler(), LinearRegression()), "gbm": GBM}
model = MODELS[MODEL]

# ④ 학습 데이터 안에서만 5-Fold 예측(OOF) 만들기 -> 여유분 k 는 여기서만 고른다
cv = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xtr, pd.qcut(ytr, 10, labels=False)))
oof = cross_val_predict(model, Xtr, ytr, cv=cv)

# ⑤ k 를 -30% ~ +60% 로 바꾸며 1인당 비용 계산 (행렬 한 번에)
ks = np.round(np.arange(-0.30, 0.6001, 0.01), 2)
quote = oof[None, :] * (1 + ks[:, None])
under = np.clip(ytr.values[None, :] - quote, 0, None).sum(axis=1)
over = np.clip(quote - ytr.values[None, :], 0, None).sum(axis=1)
cost_train = (COST_UNDER * under + COST_OVER * over) / len(ytr)
k_best = ks[np.argmin(cost_train)]
print("best k (train OOF):", k_best)

# ⑥ 고른 k 를 시험 데이터에 한 번만 적용
pred = model.fit(Xtr, ytr).predict(Xte)
q0 = pred
q1 = pred * (1 + k_best)
print("test cost k=0  :", round((COST_UNDER * np.clip(yte - q0, 0, None).sum() + COST_OVER * np.clip(q0 - yte, 0, None).sum()) / len(yte)))
print("test cost k=best:", round((COST_UNDER * np.clip(yte - q1, 0, None).sum() + COST_OVER * np.clip(q1 - yte, 0, None).sum()) / len(yte)))
print("under-estimated share k=0 / k=best:", round(np.mean(q0 < yte), 3), "/", round(np.mean(q1 < yte), 3))

# ⑦ 그림: 학습 OOF 비용 곡선
import matplotlib.pyplot as plt
plt.plot(ks, cost_train)
plt.axvline(k_best, color="black", linestyle="--")
plt.title("Cost per person vs margin k (train OOF)")
plt.xlabel("k")
plt.ylabel("cost ($)")
plt.show()

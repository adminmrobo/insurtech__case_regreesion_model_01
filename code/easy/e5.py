# AI(인공지능) x 비즈니스 #1 인슈어테크_03_회귀모델 · 초보용 코드 e5.py
# 5편 「최고 모델을 흉내 내는 나무」
# 함수 · 반복문 · if 문 · 들여쓰기 없이 위에서 아래로 읽습니다. 맨 위 [다이얼] 줄만 바꿔 다시 실행하세요.

# ① 다이얼
LEAVES = 6     # [다이얼] 대리모델 나무의 잎(규칙) 수: 2 ~ 16

# ② 데이터 읽기 · 나누기 (1부 공통: 10분위 층화 80:20)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, export_text
df = pd.read_csv("https://raw.githubusercontent.com/fintech-data/Revolution/main/data/insurance.csv").drop_duplicates()
X = pd.get_dummies(df.drop(columns="charges"), drop_first=True).astype(float)
y = df["charges"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=pd.qcut(y, 10, labels=False), random_state=42)

# ③ 선생님 모델(최종 GBM) 학습
gbm = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42).fit(Xtr, ytr)
teacher_train = gbm.predict(Xtr)
teacher_test = gbm.predict(Xte)

# ④ 대리모델: 정답 대신 "선생님의 학습 데이터 예측"을 배운다
sur = DecisionTreeRegressor(max_leaf_nodes=LEAVES, random_state=42).fit(Xtr, teacher_train)
ps = sur.predict(Xte)
print("fidelity R2 (vs GBM, test):", round(1 - ((teacher_test - ps) ** 2).sum() / ((teacher_test - teacher_test.mean()) ** 2).sum(), 4))
print("fidelity MAE (vs GBM, test):", round(np.abs(teacher_test - ps).mean()))
print("MAE vs actual (test):", round(np.abs(yte - ps).mean()), " GBM:", round(np.abs(yte - teacher_test).mean()))

# ⑤ 같은 크기로 실제 charges 에 직접 배운 나무
direct = DecisionTreeRegressor(max_leaf_nodes=LEAVES, random_state=42).fit(Xtr, ytr)
print("direct tree MAE (test):", round(np.abs(yte - direct.predict(Xte)).mean()))

# ⑥ 규칙표 (종이 한 장)
print(export_text(sur, feature_names=list(X.columns), decimals=2))

# AI(인공지능) x 비즈니스 #1 인슈어테크_03_회귀모델 · 초보용 코드 e2.py
# 2편 「지우지 말고 지킨다: 중복 · 이상치 · 로그」
# 함수 · 반복문 · if 문 · 들여쓰기 없이 위에서 아래로 읽습니다. 맨 위 [다이얼] 줄만 바꿔 다시 실행하세요.

# ① 다이얼
IQR_K = 1.5          # [다이얼] IQR 가중치 (1.0 ~ 3.0). 클수록 이상치로 덜 잡는다
TOP_SHARE = 0.10     # [다이얼] 고액으로 볼 비율 (0.10 = 상위 10%)

# ② 데이터 읽기 · 중복 확인
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
raw = pd.read_csv("https://raw.githubusercontent.com/fintech-data/Revolution/main/data/insurance.csv")
print("duplicate rows:")
print(raw[raw.duplicated(keep=False)])
print("missing values:", raw.isna().sum().sum())
df = raw.drop_duplicates().reset_index(drop=True)
print("rows after dropping the duplicate:", len(df))

# ③ 이상치 탐지 (전체 데이터에서 세기만)
q1 = df["charges"].quantile(0.25)
q3 = df["charges"].quantile(0.75)
upper = q3 + IQR_K * (q3 - q1)
flag = df["charges"] > upper
print("upper fence:", round(upper), " flagged rows:", flag.sum(), " smoker share among flagged:", round((df["smoker"][flag] == "yes").mean(), 3))

# ④ 학습/시험 나누기 (1편과 같음) · 원-핫
X = pd.get_dummies(df.drop(columns="charges"), columns=["sex", "smoker", "region"], drop_first=True).astype(float)
y = df["charges"]
band = pd.qcut(y, 10, labels=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=band, random_state=42)
top = (y_test >= y_test.quantile(1 - TOP_SHARE)).values

# ⑤ 학습 데이터의 울타리 (학습 데이터로만 계산)
t1 = y_train.quantile(0.25)
t3 = y_train.quantile(0.75)
fence = t3 + IQR_K * (t3 - t1)
keep_rows = y_train <= fence
print("train fence:", round(fence), " train rows removed by delete:", (~keep_rows).sum())

# ⑥ 같은 모델 세 번: 삭제 / 상한 고정 / 그대로
model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
p_delete = model.fit(X_train[keep_rows], y_train[keep_rows]).predict(X_test)
p_cap = model.fit(X_train, y_train.clip(upper=fence)).predict(X_test)
p_keep = model.fit(X_train, y_train).predict(X_test)
p_log = np.exp(model.fit(X_train, np.log(y_train)).predict(X_test))

# ⑦ 손대지 않은 시험 데이터에서 비교
yt = y_test.values
mae = [np.abs(yt - p_delete).mean(), np.abs(yt - p_cap).mean(), np.abs(yt - p_keep).mean(), np.abs(yt - p_log).mean()]
mae_top = [np.abs(yt - p_delete)[top].mean(), np.abs(yt - p_cap)[top].mean(), np.abs(yt - p_keep)[top].mean(), np.abs(yt - p_log)[top].mean()]
under = [(p_delete < yt).mean(), (p_cap < yt).mean(), (p_keep < yt).mean(), (p_log < yt).mean()]
result = pd.DataFrame({"MAE": mae, "MAE_top": mae_top, "under_rate": under}, index=["delete", "cap", "keep", "keep + log"])
print(result.round(3))

# ⑧ 분포 그림: 원래 금액과 로그 금액
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
df["charges"].plot.hist(bins=40, ax=axes[0], title="charges (skew %.2f)" % df["charges"].skew())
np.log(df["charges"]).plot.hist(bins=40, ax=axes[1], title="log(charges) (skew %.2f)" % np.log(df["charges"]).skew())
plt.show()

# AI(인공지능) x 비즈니스 #1 인슈어테크_03_회귀모델 · 초보용 코드 e3.py
# 3편 「모델 다섯, 곡선 다섯」
# 함수 · 반복문 · if 문 · 들여쓰기 없이 위에서 아래로 읽습니다. 맨 위 [다이얼] 줄만 바꿔 다시 실행하세요.

# ① 다이얼
TREE_DEPTH = 4       # [다이얼] 의사결정나무 깊이 (1 ~ 10)
KNN_K = 10           # [다이얼] KNN 이웃 수 (1 ~ 100)
GBM_TREES = 200      # [다이얼] 그래디언트 부스팅 나무 개수 (1 ~ 400)
RIDGE_ALPHA = 1.0    # [다이얼] 릿지 규제 세기 (0.001 ~ 10000)

# ② 데이터 읽기 · 같은 행 1건 빼기 · 원-핫 · 나누기 (1편과 같음)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
df = pd.read_csv("https://raw.githubusercontent.com/fintech-data/Revolution/main/data/insurance.csv").drop_duplicates().reset_index(drop=True)
X = pd.get_dummies(df.drop(columns="charges"), columns=["sex", "smoker", "region"], drop_first=True).astype(float)
y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=pd.qcut(y, 10, labels=False), random_state=42)

# ③ 모델 다섯 학습 (표준화가 필요한 모델은 Pipeline 안에서 학습 데이터에만 fit)
p_lin = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train).predict(X_test)
p_ridge = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(X_train, y_train).predict(X_test)
p_tree = DecisionTreeRegressor(max_depth=TREE_DEPTH, random_state=42).fit(X_train, y_train).predict(X_test)
p_knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=KNN_K)).fit(X_train, y_train).predict(X_test)
p_gbm = GradientBoostingRegressor(n_estimators=GBM_TREES, max_depth=3, learning_rate=0.05, random_state=42).fit(X_train, y_train).predict(X_test)
p_base = np.full(len(y_test), y_train.mean())

# ④ 시험 데이터 268명에서 비교 (기준선 포함)
yt = y_test.values
top = yt >= np.quantile(yt, 0.9)
names = ["Baseline", "Linear", "Ridge", "Decision Tree", "KNN", "Gradient Boosting"]
mae = [np.abs(yt - p_base).mean(), np.abs(yt - p_lin).mean(), np.abs(yt - p_ridge).mean(), np.abs(yt - p_tree).mean(), np.abs(yt - p_knn).mean(), np.abs(yt - p_gbm).mean()]
mae_top = [np.abs(yt - p_base)[top].mean(), np.abs(yt - p_lin)[top].mean(), np.abs(yt - p_ridge)[top].mean(), np.abs(yt - p_tree)[top].mean(), np.abs(yt - p_knn)[top].mean(), np.abs(yt - p_gbm)[top].mean()]
under = [(p_base < yt).mean(), (p_lin < yt).mean(), (p_ridge < yt).mean(), (p_tree < yt).mean(), (p_knn < yt).mean(), (p_gbm < yt).mean()]
print(pd.DataFrame({"MAE": mae, "MAE_top10": mae_top, "under_rate": under}, index=names).round(3))

# ⑤ 교육용 그림: 나이 하나만 쓴 나무의 예측 곡선 (깊이 다이얼)
import matplotlib.pyplot as plt
grid = pd.DataFrame({"age": np.arange(18, 64.5, 0.5)})
curve = DecisionTreeRegressor(max_depth=TREE_DEPTH, random_state=42).fit(X_train[["age"]], y_train).predict(grid)
ax = df.plot.scatter(x="age", y="charges", s=6, color="lightgray")
ax.plot(grid["age"], curve, color="tab:purple")
ax.set_title("Age-only tree, depth %d (teaching model)" % TREE_DEPTH)
plt.show()

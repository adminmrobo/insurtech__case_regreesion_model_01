# AI(인공지능) x 비즈니스 #1 인슈어테크_03_회귀모델 · 초보용 코드 e6.py
# 6편 「기준서가 곧 견적: 의료비 견적 앱」
# 함수 · 반복문 · if 문 · 들여쓰기 없이 위에서 아래로 읽습니다. 맨 위 [다이얼] 줄만 바꿔 다시 실행하세요.

# ① 다이얼: 견적을 낼 고객
AGE = 45          # [다이얼] 나이 18 ~ 64
BMI = 31.0        # [다이얼] 체질량지수
CHILDREN = 1      # [다이얼] 자녀 수
SEX = "female"    # [다이얼] "female" 또는 "male"
SMOKER = "yes"    # [다이얼] "yes" 또는 "no"
REGION = "southeast"  # [다이얼] northeast · northwest · southeast · southwest
MARGIN_K = 0.06   # [다이얼] 여유분 k (6편: 학습 데이터 5-Fold 로 고른 값 +0.06)

# ② 데이터 · 나누기 · 선생님 모델 · 대리모델 (5편과 같음, 잎 6개)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
df = pd.read_csv("https://raw.githubusercontent.com/fintech-data/Revolution/main/data/insurance.csv").drop_duplicates()
X = pd.get_dummies(df.drop(columns="charges"), drop_first=True).astype(float)
y = df["charges"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=pd.qcut(y, 10, labels=False), random_state=42)
gbm = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42).fit(Xtr, ytr)
sur = DecisionTreeRegressor(max_leaf_nodes=6, random_state=42).fit(Xtr, gbm.predict(Xtr))

# ③ 고객 한 명 견적 (열 순서는 학습 데이터와 같게)
row = {"age": AGE, "bmi": BMI, "children": CHILDREN, "sex_male": float(SEX == "male"), "smoker_yes": float(SMOKER == "yes")}
row["region_northwest"] = float(REGION == "northwest")
row["region_southeast"] = float(REGION == "southeast")
row["region_southwest"] = float(REGION == "southwest")
one = pd.DataFrame([row])[X.columns]
quote = sur.predict(one)[0]
print("rule-table quote:", round(quote, 2))
print("with margin k:", round(quote * (1 + MARGIN_K), 2))
print("original GBM:", round(gbm.predict(one)[0], 2))

# ④ 시험 268명 일괄 견적: 실제보다 $10,000 넘게 적게 잡은 사람 수
q = sur.predict(Xte) * (1 + MARGIN_K)
print("test under-estimated by more than $10,000:", int(((yte - q) > 10000).sum()), "of", len(yte))

# ⑤ 공정성: 성별만 바꾸면 견적이 바뀌는 사람 수
flip = Xte.assign(sex_male=1 - Xte["sex_male"])
print("quotes changed by flipping sex - rule table:", int((np.abs(sur.predict(flip) - sur.predict(Xte)) > 0.01).sum()))
print("quotes changed by flipping sex - GBM:", int((np.abs(gbm.predict(flip) - gbm.predict(Xte)) > 1).sum()))

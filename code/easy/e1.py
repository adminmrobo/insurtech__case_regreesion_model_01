# AI(인공지능) x 비즈니스 #1 인슈어테크_03_회귀모델 · 초보용 코드 e1.py
# 1편 「모두에게 평균만: 평균의 함정」
# 함수 · 반복문 · if 문 · 들여쓰기 없이 위에서 아래로 읽습니다. 맨 위 [다이얼] 줄만 바꿔 다시 실행하세요.

# ① 다이얼
RULE_COLUMN = "smoker"   # [다이얼] 규칙에 쓸 열: "smoker" 또는 "sex" 또는 "region"
TOP_SHARE = 0.10         # [다이얼] 고액으로 볼 비율 (0.10 = 상위 10%)

# ② 데이터 읽기 · 같은 행 1건 빼기
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("https://raw.githubusercontent.com/fintech-data/Revolution/main/data/insurance.csv")
df = df.drop_duplicates().reset_index(drop=True)
print("rows:", len(df))

# ③ 학습/시험 나누기 (금액 10분위로 층화, 80:20)
band = pd.qcut(df["charges"], 10, labels=False)
train, test = train_test_split(df, test_size=0.2, stratify=band, random_state=42)
print("train:", len(train), " test:", len(test))

# ④ 기준선: 모두에게 학습 데이터 평균만 답하기
base_pred = np.full(len(test), train["charges"].mean())
top = test["charges"] >= test["charges"].quantile(1 - TOP_SHARE)
print("baseline MAE:", round(np.abs(test["charges"] - base_pred).mean()))
print("baseline MAE on top charges:", round(np.abs(test["charges"][top] - base_pred[top.values]).mean()))
print("baseline R2:", round(1 - ((test["charges"] - base_pred) ** 2).sum() / ((test["charges"] - test["charges"].mean()) ** 2).sum(), 4))
print("sum gap (pred - actual) / actual:", round((base_pred.sum() - test["charges"].sum()) / test["charges"].sum(), 4))

# ⑤ 규칙: 다이얼 열의 값마다 학습 데이터 평균으로 답하기
group_mean = train.groupby(RULE_COLUMN)["charges"].mean()
rule_pred = test[RULE_COLUMN].map(group_mean).values
print(group_mean.round(0))
print("rule MAE:", round(np.abs(test["charges"] - rule_pred).mean()))
print("rule MAE on top charges:", round(np.abs(test["charges"][top] - rule_pred[top.values]).mean()))
print("rule R2:", round(1 - ((test["charges"] - rule_pred) ** 2).sum() / ((test["charges"] - test["charges"].mean()) ** 2).sum(), 4))

# ⑥ 그림: 나이 x 의료비, 다이얼 열로 색칠
import matplotlib.pyplot as plt
ax = df.plot.scatter(x="age", y="charges", c=(df[RULE_COLUMN] == df[RULE_COLUMN].iloc[0]).map({True: "tab:orange", False: "tab:blue"}), s=8)
ax.axhline(train["charges"].mean(), color="black", linestyle="--")
ax.set_title("Age x charges (dashed = train mean)")
plt.show()

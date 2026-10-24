"""
AI(인공지능) x 비즈니스 #1 인슈어테크_03_회귀모델 · 정식 코드 f6.py
6편 「기준서가 곧 견적: 의료비 견적 앱」
무엇을, 어떤 순서로:
 ① data/surrogate_tree.json (5편의 대리모델 규칙표)을 읽는다 -> 없으면 f5 를 실행해 만든다
 ② JSON 규칙을 위에서부터 따라가는 "앱과 같은 계산"으로 시험 268명 견적을 내고, sklearn 대리모델 예측과 대조 (불일치 수 · 최대 차이)
 ③ 배포할 규칙표에 붙일 여유분 k: 학습 데이터 5-Fold 안에서 (GBM -> 대리나무) 를 통째로 다시 학습한 OOF 견적으로 3:1 비용 최소 k 를 고른다
 ④ 시험 데이터에 한 번 적용: 비용 · 과소 추정 비율 · 크게 적게 잡은 사람($10,000 넘게)
 ⑤ 공정성 점검: 성별 · 지역을 바꾸면 견적이 바뀌는 사람 수 (규칙표 vs 원 모델 GBM)
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
import common as C

CU, CO = 3.0, 1.0
KS = np.round(np.arange(-0.30, 0.6001, 0.01), 2)
BIG_UNDER = 10000.0


def load_tree():
    if not os.path.exists("data/surrogate_tree.json"):
        import f5
        f5.run(verbose=False)
    with open("data/surrogate_tree.json", encoding="utf-8") as f:
        return json.load(f)


def quote_one(nodes, row):
    """앱(JS)과 똑같은 계산: 뿌리에서 시작해 '<= 기준이면 왼쪽' 으로 내려간다."""
    i = 0
    path = []
    while nodes[i]["left"] is not None:
        n = nodes[i]
        go_left = row[n["feature"]] <= n["threshold"]
        path.append((n["feature"], n["threshold"], go_left))
        i = n["left"] if go_left else n["right"]
    return nodes[i]["value"], path


def cost(y, q):
    y, q = np.asarray(y, float), np.asarray(q, float)
    return float((CU * np.clip(y - q, 0, None).sum() + CO * np.clip(q - y, 0, None).sum()) / len(y))


def run(verbose=True):
    df, X, y, info = C.load_clean_xy()
    Xtr, Xte, ytr, yte = C.split(X, y)
    T = load_tree()
    nodes, L = T["nodes"], T["leaves"]

    # ② JSON 규칙 vs sklearn
    gbm = C.final_model().fit(Xtr, ytr)
    sur = DecisionTreeRegressor(max_leaf_nodes=L, random_state=C.SEED).fit(Xtr, gbm.predict(Xtr))
    sk = sur.predict(Xte)
    app = np.array([quote_one(nodes, r)[0] for _, r in Xte.iterrows()])
    diff = np.abs(app - sk)
    mismatch = int((diff > 0.01).sum())   # JSON 은 소수 둘째 자리로 저장 -> 1센트 넘게 다르면 불일치

    # ③ 여유분 k: 학습 데이터 안의 5-Fold 로 (GBM -> 대리나무) 전체를 다시 학습
    cv = list(StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(Xtr, C.strata(ytr)))
    oof = np.zeros(len(ytr))
    for a, b in cv:
        g = C.final_model().fit(Xtr.iloc[a], ytr.iloc[a])
        s = DecisionTreeRegressor(max_leaf_nodes=L, random_state=C.SEED).fit(Xtr.iloc[a], g.predict(Xtr.iloc[a]))
        oof[b] = s.predict(Xtr.iloc[b])
    cs = [cost(ytr, oof * (1 + k)) for k in KS]
    k_best = float(KS[int(np.argmin(cs))])

    # ④ 시험에 한 번
    q0, q1 = app, app * (1 + k_best)
    gb = gbm.predict(Xte)
    res = pd.DataFrame({
        "quote (k=0)": [cost(yte, q0), np.mean(q0 < yte), np.mean(np.abs(yte - q0)), int(((yte - q0) > BIG_UNDER).sum())],
        "quote x (1+k)": [cost(yte, q1), np.mean(q1 < yte), np.mean(np.abs(yte - q1)), int(((yte - q1) > BIG_UNDER).sum())],
        "GBM (original)": [cost(yte, gb), np.mean(gb < yte), np.mean(np.abs(yte - gb)), int(((yte - gb) > BIG_UNDER).sum())],
    }, index=["cost_3to1", "under_rate", "MAE", "under_by_10k+"])

    # ⑤ 공정성: 성별 · 지역 바꾸기
    flip_sex = Xte.copy()
    flip_sex["sex_male"] = 1 - flip_sex["sex_male"]
    sur_sex = int((np.abs(sur.predict(flip_sex) - sk) > 0.01).sum())
    gbm_sex_d = np.abs(gbm.predict(flip_sex) - gb)
    flip_reg = Xte.copy()
    flip_reg[["region_northwest", "region_southeast", "region_southwest"]] = 0.0   # 모두 northeast 로
    sur_reg = int((np.abs(sur.predict(flip_reg) - sk) > 0.01).sum())
    gbm_reg_d = np.abs(gbm.predict(flip_reg) - gb)
    fair = dict(sur_sex=sur_sex, gbm_sex=int((gbm_sex_d > 1).sum()), gbm_sex_max=float(gbm_sex_d.max()),
                sur_region=sur_reg, gbm_region=int((gbm_reg_d > 1).sum()), gbm_region_max=float(gbm_reg_d.max()))

    out = dict(nodes=nodes, L=L, used=T["features_used"], app=app, sk=sk, gb=gb, mismatch=mismatch, max_diff=float(diff.max()),
               k_best=k_best, k_curve=cs, res=res, fair=fair, Xte=Xte, yte=yte)
    if verbose:
        print("surrogate rules: %d leaves, features used: %s" % (L, T["features_used"]))
        print("JSON rules (app) vs sklearn surrogate on test %d: mismatch=%d  max|diff|=%.4f" % (len(yte), mismatch, diff.max()))
        print("margin k for the rule table, chosen on TRAIN 5-fold (GBM->tree refit per fold), cost 3:1: %+.2f" % k_best)
        print(res.round(3).to_string())
        print()
        print("fairness: quotes changed when sex is flipped  -> rule table %d / GBM %d (max change $%.0f)" %
              (fair["sur_sex"], fair["gbm_sex"], fair["gbm_sex_max"]))
        print("fairness: quotes changed when region -> northeast -> rule table %d / GBM %d (max change $%.0f)" %
              (fair["sur_region"], fair["gbm_region"], fair["gbm_region_max"]))
    return out


if __name__ == "__main__":
    run()

"""
AI(인공지능) x 비즈니스 #1 인슈어테크_03_회귀모델 · 정식 코드 f4.py
4편 「여유분과 비용 실험실」
무엇을, 어떤 순서로:
 ① 데이터를 읽고 정리, 1부 나누기(학습 1,069 / 시험 268)
 ② 모델 세 개(최종 GBM · 선형 회귀 · 흡연 규칙)의 학습 데이터 5-Fold OOF 예측과 시험 예측을 만든다
 ③ 견적 = 예측 × (1 + k). 비용 = 3 × 과소 금액 + 1 × 과대 금액 (1인당 평균)
 ④ 학습 OOF 로만 최적 k 를 고른다 -> 시험에 한 번 적용
 ⑤ (반칙 비교) 시험 데이터로 고른 k
 ⑥ 이론: 3:1 비용이면 최적 견적은 75% 분위수 -> GBM quantile(alpha=0.75) 과 비교
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
import common as C

CU, CO = 3.0, 1.0                      # 과소 1달러 : 과대 1달러 = 3 : 1
KS = np.round(np.arange(-0.30, 0.6001, 0.01), 2)


def cost(y, q, cu=CU, co=CO):
    """1인당 평균 비용."""
    y = np.asarray(y, float)
    q = np.asarray(q, float)
    under = np.clip(y - q, 0, None)
    over = np.clip(q - y, 0, None)
    return float((cu * under.sum() + co * over.sum()) / len(y))


def best_k(y, p, cu=CU, co=CO):
    cs = np.array([cost(y, p * (1 + k), cu, co) for k in KS])
    i = int(np.argmin(cs))
    return float(KS[i]), cs


def smoker_rule(Xa, ya, Xb):
    m = ya.groupby(Xa["smoker_yes"]).mean()
    return Xb["smoker_yes"].map(m).astype(float).values


def run(verbose=True):
    df, X, y, info = C.load_clean_xy()
    Xtr, Xte, ytr, yte = C.split(X, y)
    cv = list(StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(Xtr, C.strata(ytr)))
    ms = C.models()
    oof, te = {}, {}
    for name, key in [("Gradient Boosting", "gbm"), ("Linear", "lin")]:
        oof[key] = cross_val_predict(ms[name], Xtr, ytr, cv=cv)
        te[key] = ms[name].fit(Xtr, ytr).predict(Xte)
    o = np.zeros(len(ytr))
    for a, b in cv:
        o[b] = smoker_rule(Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b])
    oof["smk"] = o
    te["smk"] = smoker_rule(Xtr, ytr, Xte)

    rows = []
    for key in ["gbm", "lin", "smk"]:
        k_cv, _ = best_k(ytr.values, oof[key])
        k_te, _ = best_k(yte.values, te[key])
        p0, p1 = te[key], te[key] * (1 + k_cv)
        rows.append(dict(model=key, k_train_cv=k_cv, k_test_cheat=k_te,
                         cost_k0=cost(yte, p0), cost_kcv=cost(yte, p1), cost_kte=cost(yte, te[key] * (1 + k_te)),
                         under_rate_k0=float(np.mean(p0 < yte)), under_rate_kcv=float(np.mean(p1 < yte)),
                         mae_k0=float(np.mean(np.abs(yte - p0))), mae_kcv=float(np.mean(np.abs(yte - p1)))))
    tab = pd.DataFrame(rows).set_index("model")

    # 비용 비율에 따른 최적 k (학습 OOF, 최종 모델)
    ratio_k = {r: best_k(ytr.values, oof["gbm"], r, 1.0)[0] for r in range(1, 11)}

    # 이론: 75% 분위수 GBM
    qm = GradientBoostingRegressor(loss="quantile", alpha=CU / (CU + CO), **{k: v for k, v in C.FINAL_PARAMS.items()})
    qp = qm.fit(Xtr, ytr).predict(Xte)
    quant = dict(alpha=CU / (CU + CO), cost=cost(yte, qp), under_rate=float(np.mean(qp < yte)),
                 mae=float(np.mean(np.abs(yte - qp))))

    out = dict(oof=oof, te=te, ytr=ytr, yte=yte, Xte=Xte, Xtr=Xtr, tab=tab, ratio_k=ratio_k, quant=quant, qpred=qp)
    if verbose:
        print("cost = %.0f x under$ + %.0f x over$  (per person, test n=%d)" % (CU, CO, len(yte)))
        print("k chosen on TRAIN 5-fold OOF only; k_test_cheat = chosen on test (not allowed)")
        print(tab.round({"cost_k0": 0, "cost_kcv": 0, "cost_kte": 0, "under_rate_k0": 3, "under_rate_kcv": 3,
                         "mae_k0": 0, "mae_kcv": 0}).to_string())
        print()
        print("final model: best k by cost ratio (train OOF)")
        print("  " + "  ".join("%d:1->%+.2f" % (r, k) for r, k in ratio_k.items()))
        print()
        print("quantile GBM alpha=%.2f (test): cost=%.0f  under_rate=%.3f  MAE=%.0f" %
              (quant["alpha"], quant["cost"], quant["under_rate"], quant["mae"]))
    return out


if __name__ == "__main__":
    run()

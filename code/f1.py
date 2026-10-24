"""
AI(인공지능) x 비즈니스 #1 인슈어테크_03_회귀모델 · 정식 코드 f1.py
1편 「모두에게 평균만: 평균의 함정」
무엇을, 어떤 순서로:
 ① 데이터를 읽고 정리한다(common.py) -> 1,337행
 ② 미리 정한 학습/시험 나누기를 쓴다 (charges 10분위 층화 80:20, random_state=42)
 ③ 기준선 "모두에게 학습 평균"과 사람이 떠올릴 규칙 두 개, 4편 최종 모델을 같은 시험 데이터에서 잰다
 ④ MAE · RMSE · R² · 고액(상위 10%) MAE · 과소 추정 비율 · 예측 합계 차이를 한 표로 본다
 ⑤ 숨은 열쇠: 흡연 여부별 평균 의료비
"""
import numpy as np
import pandas as pd
import common as C


def run(verbose=True):
    df, X, y, info = C.load_clean_xy()
    Xtr, Xte, ytr, yte = C.split(X, y)
    tr_smk = Xtr["smoker_yes"] == 1
    m_smk, m_non = ytr[tr_smk].mean(), ytr[~tr_smk].mean()
    age_bins = [17, 29, 39, 49, 59, 64]
    tr_age = pd.cut(Xtr["age"], age_bins)
    age_mean = ytr.groupby(tr_age, observed=False).mean()
    te_age = pd.cut(Xte["age"], age_bins)

    preds = {
        "Baseline (train mean)": C.baseline(ytr, len(yte)),
        "Rule: smoker mean": np.where(Xte["smoker_yes"] == 1, m_smk, m_non),
        "Rule: age band mean": te_age.map(age_mean).astype(float).values,
    }
    gbm = C.final_model().fit(Xtr, ytr)
    preds["Final model (ep4)"] = gbm.predict(Xte)

    rows = []
    for name, p in preds.items():
        m = C.metrics(yte, p)
        m["sum_gap"] = (p.sum() - yte.sum()) / yte.sum()
        rows.append(dict(rule=name, **m))
    tab = pd.DataFrame(rows).set_index("rule")
    smoke = df.groupby("smoker")[C.TARGET].agg(["count", "mean", "median"])
    out = dict(info=info, n_train=len(ytr), n_test=len(yte), train_mean=float(ytr.mean()),
               test_mean=float(yte.mean()), smoker_mean=float(m_smk), nonsmoker_mean=float(m_non),
               age_mean={str(k): float(v) for k, v in age_mean.items()}, table=tab, smoke=smoke,
               preds=preds, yte=yte, Xte=Xte, Xtr=Xtr, ytr=ytr, df=df)
    if verbose:
        print("rows=%d  (raw %d, dropped duplicate %d)" % (info["rows_clean"], info["rows_raw"], len(info["dropped_duplicate_rows"])))
        print("charges mean=%.0f  median=%.0f  max=%.0f" % (df.charges.mean(), df.charges.median(), df.charges.max()))
        print("train=%d  test=%d  train mean=%.2f" % (len(ytr), len(yte), ytr.mean()))
        print(tab.round({"MAE": 0, "RMSE": 0, "R2": 4, "MAE_top10": 0, "under_rate": 3, "bias": 0, "sum_gap": 4}).to_string())
        print()
        print(smoke.round(0).to_string())
        print("smoker / non-smoker mean ratio = %.2f" % (df[df.smoker == "yes"].charges.mean() / df[df.smoker == "no"].charges.mean()))
    return out


if __name__ == "__main__":
    run()

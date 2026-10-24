"""
AI(인공지능) x 비즈니스 #1 인슈어테크_03_회귀모델 · 정식 코드 f2.py
2편 「지우지 말고 지킨다: 중복 · 이상치 · 로그」
무엇을, 어떤 순서로:
 ① 완전히 같은 행(중복)을 찾고, 남겨 두면 학습과 시험에 동시에 들어가는지 확인한다
 ② 결측 개수를 센다
 ③ IQR 가중치(1.0 ~ 3.0)별로 이상치로 걸리는 행 수와 그중 흡연자 비율을 센다
 ④ 학습 데이터에만 네 가지 처리(삭제 · 상한 고정 · 그대로 · 그대로 + 로그)를 하고
    같은 모델(선형 · 최종 모델)을 학습해, 손대지 않은 시험 268명에서 비교한다
 ⑤ charges 와 log(charges) 의 왜도
 ⑥ 정리 규칙별 바뀐 행 수
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import common as C

KS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
METHODS = ["delete", "cap", "keep", "keep_log"]


def iqr_fence(s, k):
    q1, q3 = s.quantile([0.25, 0.75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


def treat(Xtr, ytr, method, k=1.5):
    """학습 데이터에만 처리. 기준(울타리)도 학습 데이터로만 계산한다."""
    lo, hi = iqr_fence(ytr, k)
    if method == "delete":
        keep = (ytr >= lo) & (ytr <= hi)
        return Xtr[keep], ytr[keep], False
    if method == "cap":
        return Xtr, ytr.clip(lo, hi), False
    if method == "keep_log":
        return Xtr, ytr, True
    return Xtr, ytr, False


def run(verbose=True):
    raw = C.load_raw()
    df, info = C.clean(raw)
    X, y = C.encode(df)
    Xtr, Xte, ytr, yte = C.split(X, y)

    # ① 중복: 남겨 두었다면 어디로 갔을까 (같은 규칙으로 1,338행을 나눠 본다)
    dup = raw[raw.duplicated(keep=False)]
    Xr, yr = C.encode(raw)
    a, b, _, _ = train_test_split(Xr, yr, test_size=0.2, stratify=C.strata(yr), random_state=C.SEED)
    dup_sets = ["train" if i in a.index else "test" for i in dup.index]

    # ② 결측
    missing = int(raw.isna().sum().sum())

    # ③ IQR 가중치별 탐지 (전체 1,337행, 탐지는 세기만 한다)
    det = []
    for k in KS:
        lo, hi = iqr_fence(df[C.TARGET], k)
        m = (df[C.TARGET] < lo) | (df[C.TARGET] > hi)
        blo, bhi = iqr_fence(df["bmi"], k)
        bm = (df["bmi"] < blo) | (df["bmi"] > bhi)
        det.append(dict(k=k, charges_fence=hi, charges_flagged=int(m.sum()),
                        smoker_share=float((df.loc[m, "smoker"] == "yes").mean()) if m.sum() else 0.0,
                        bmi_flagged=int(bm.sum()), bmi_fence=bhi))
    det = pd.DataFrame(det).set_index("k")

    # ④ 처리 방법 비교 (학습에만 처리 -> 시험 268명은 그대로)
    rows, preds = [], {}
    for mname in ["Linear", "Gradient Boosting"]:
        for meth in METHODS:
            xt, yt, use_log = treat(Xtr, ytr, meth)
            mdl = C.models()[mname]
            mdl.fit(xt, np.log(yt) if use_log else yt)
            p = mdl.predict(Xte)
            p = np.exp(p) if use_log else p
            mt = C.metrics(yte, p)
            mt.update(model=mname, method=meth, train_rows=len(yt), train_mean=float(yt.mean()),
                      train_max=float(yt.max()), pred_mean=float(p.mean()))
            rows.append(mt)
            preds[(mname, meth)] = p
    comp = pd.DataFrame(rows).set_index(["model", "method"])
    lo, hi = iqr_fence(ytr, 1.5)
    n_del = int(((ytr < lo) | (ytr > hi)).sum())
    del_smk = float(Xtr.loc[(ytr < lo) | (ytr > hi), "smoker_yes"].mean())

    # ⑤ 왜도
    sk = dict(raw=float(df[C.TARGET].skew()), log=float(np.log(df[C.TARGET]).skew()),
              mean=float(df[C.TARGET].mean()), median=float(df[C.TARGET].median()),
              exp_mean_log=float(np.exp(np.log(df[C.TARGET]).mean())))

    out = dict(info=info, dup=dup, dup_sets=dup_sets, missing=missing, det=det, comp=comp, preds=preds,
               train_fence=float(hi), n_del=n_del, del_smoker_share=del_smk, skew=sk,
               df=df, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)
    if verbose:
        print("① duplicate rows in raw data:")
        print(dup.to_string())
        print("if kept, the two copies go to:", dup_sets)
        print("② missing values:", missing)
        print("③ IQR detection on all %d rows" % len(df))
        print(det.round({"charges_fence": 0, "smoker_share": 3, "bmi_fence": 2}).to_string())
        print("④ treatments on TRAIN only (train IQR x1.5 upper fence = %.0f, %d train rows flagged, smokers %.1f%%)"
              % (hi, n_del, del_smk * 100))
        print(comp[["train_rows", "train_mean", "train_max", "MAE", "MAE_top10", "R2", "under_rate", "bias"]]
              .round({"train_mean": 0, "train_max": 0, "MAE": 0, "MAE_top10": 0, "R2": 4, "under_rate": 3, "bias": 0}).to_string())
        print("⑤ skewness charges=%.3f  log(charges)=%.3f" % (sk["raw"], sk["log"]))
        print("   mean=%.0f  median=%.0f  exp(mean(log))=%.0f" % (sk["mean"], sk["median"], sk["exp_mean_log"]))
        print("⑥ cleaning rules")
        C.clean(raw, verbose=True)
    return out


if __name__ == "__main__":
    run()

"""
AI(인공지능) x 비즈니스 #1 인슈어테크_03_회귀모델 · 정식 코드 f3.py
3편 「모델 다섯, 곡선 다섯」
무엇을, 어떤 순서로:
 ① (교육용) 나이 하나만 쓰는 다섯 모델을 복잡도 8단계로 학습해 예측 곡선을 만든다
    선형(항상 직선) · 릿지(8차 다항, alpha 큰 값 -> 작은 값) · 나무(깊이 1 -> 8) · KNN(k 500 -> 1) · GBM(나무 1 -> 300개)
 ② 전체 특성 8개로 common.models() 다섯 모델을 학습해 시험 268명에서 기준선과 비교한다
 ③ 최종 모델(GBM) 설정을 학습 데이터만의 5-Fold 교차검증으로 다시 고른다 (common.FINAL_PARAMS 와 같아야 한다)
 ④ 모델별 고유 특성: 선형 계수(표준화 후) · 나무 깊이별 / KNN k별 교차검증 MAE · GBM 나무 개수별 학습 곡선
"""
import itertools
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import common as C

LEVELS = 8
RIDGE_ALPHA = [1e4, 1e3, 1e2, 10, 1, 0.1, 0.01, 0.001]
TREE_DEPTH = [1, 2, 3, 4, 5, 6, 7, 8]
KNN_K = [500, 200, 100, 50, 20, 10, 5, 1]
GBM_N = [1, 5, 10, 25, 50, 100, 200, 300]
GRID = np.round(np.arange(18, 64.01, 0.5), 1)


def age_model(name, lv):
    if name == "Linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "Ridge":
        return make_pipeline(StandardScaler(), PolynomialFeatures(8, include_bias=False), StandardScaler(), Ridge(alpha=RIDGE_ALPHA[lv]))
    if name == "Decision Tree":
        return DecisionTreeRegressor(max_depth=TREE_DEPTH[lv], random_state=C.SEED)
    if name == "KNN":
        return KNeighborsRegressor(n_neighbors=KNN_K[lv])
    return GradientBoostingRegressor(n_estimators=GBM_N[lv], max_depth=3, learning_rate=0.05, random_state=C.SEED)


def train_cv(y):
    skf = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    return list(skf.split(np.zeros(len(y)), C.strata(y)))


def run(verbose=True):
    df, X, y, info = C.load_clean_xy()
    Xtr, Xte, ytr, yte = C.split(X, y)

    # ① 나이 하나 모델의 곡선
    curves, age_mae = {}, []
    a_tr, a_te, g = Xtr[["age"]], Xte[["age"]], pd.DataFrame({"age": GRID})
    for name in C.MODEL_NAMES:
        curves[name] = []
        for lv in range(LEVELS):
            m = age_model(name, lv).fit(a_tr, ytr)
            curves[name].append(np.round(m.predict(g), 1).tolist())
            age_mae.append(dict(model=name, level=lv + 1, train_MAE=float(np.mean(np.abs(m.predict(a_tr) - ytr))),
                                test_MAE=float(np.mean(np.abs(m.predict(a_te) - yte)))))
    age_mae = pd.DataFrame(age_mae)

    # ② 전체 특성 다섯 모델
    rows, preds = [], {}
    base = C.baseline(ytr, len(yte))
    rows.append(dict(model="Baseline (train mean)", **C.metrics(yte, base)))
    fitted = {}
    for name, m in C.models().items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        preds[name] = p
        fitted[name] = m
        rows.append(dict(model=name, **C.metrics(yte, p)))
    full = pd.DataFrame(rows).set_index("model")

    # ③ 최종 모델 설정 다시 고르기 (학습 데이터만)
    cv = train_cv(ytr)
    grid = []
    for ne, md, lr in itertools.product([100, 200, 300], [2, 3, 4], [0.05, 0.1]):
        p = cross_val_predict(GradientBoostingRegressor(n_estimators=ne, max_depth=md, learning_rate=lr, random_state=C.SEED), Xtr, ytr, cv=cv)
        grid.append(dict(n_estimators=ne, max_depth=md, learning_rate=lr, cv_MAE=float(np.mean(np.abs(p - ytr)))))
    grid = pd.DataFrame(grid).sort_values("cv_MAE").reset_index(drop=True)
    best = grid.iloc[0]
    same = (int(best.n_estimators), int(best.max_depth), float(best.learning_rate)) == (
        C.FINAL_PARAMS["n_estimators"], C.FINAL_PARAMS["max_depth"], C.FINAL_PARAMS["learning_rate"])

    # ④ 모델별 고유 특성
    lin = fitted["Linear"].named_steps["linearregression"]
    coef = pd.Series(lin.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)
    depth_cv = []
    for d in range(1, 11):
        p = cross_val_predict(DecisionTreeRegressor(max_depth=d, random_state=C.SEED), Xtr, ytr, cv=cv)
        depth_cv.append(dict(depth=d, cv_MAE=float(np.mean(np.abs(p - ytr)))))
    depth_cv = pd.DataFrame(depth_cv)
    k_cv = []
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]:
        p = cross_val_predict(make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k)), Xtr, ytr, cv=cv)
        k_cv.append(dict(k=k, cv_MAE=float(np.mean(np.abs(p - ytr)))))
    k_cv = pd.DataFrame(k_cv)
    n_max = 400
    tr_curve, cv_curve = np.zeros(n_max), np.zeros(n_max)
    for tr_i, va_i in cv:
        m = GradientBoostingRegressor(n_estimators=n_max, max_depth=3, learning_rate=0.05, random_state=C.SEED).fit(Xtr.iloc[tr_i], ytr.iloc[tr_i])
        for j, (pt, pv) in enumerate(zip(m.staged_predict(Xtr.iloc[tr_i]), m.staged_predict(Xtr.iloc[va_i]))):
            tr_curve[j] += np.mean(np.abs(pt - ytr.iloc[tr_i])) / len(cv)
            cv_curve[j] += np.mean(np.abs(pv - ytr.iloc[va_i])) / len(cv)
    stage = pd.DataFrame({"n_trees": np.arange(1, n_max + 1), "train_MAE": tr_curve, "cv_MAE": cv_curve})

    out = dict(curves=curves, grid_age=GRID.tolist(), age_mae=age_mae, full=full, preds=preds, grid=grid, same=same,
               coef=coef, depth_cv=depth_cv, k_cv=k_cv, stage=stage, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, df=df)
    if verbose:
        print("① age-only models: train / test MAE at complexity level 1 and 8")
        print(age_mae[age_mae.level.isin([1, 8])].pivot(index="model", columns="level", values=["train_MAE", "test_MAE"]).round(0).to_string())
        print()
        print("② all 8 features, test %d rows" % len(yte))
        print(full.round({"MAE": 0, "RMSE": 0, "R2": 4, "MAE_top10": 0, "under_rate": 3, "bias": 0}).to_string())
        print()
        print("③ GBM grid on TRAIN only (5-fold CV MAE), top 6")
        print(grid.head(6).round({"cv_MAE": 0}).to_string())
        print("best == common.FINAL_PARAMS:", same)
        print()
        print("④ linear coefficients (standardized features, $ per 1 SD)")
        print(coef.round(0).to_string())
        print("tree depth (train CV MAE):", depth_cv.round(0).set_index("depth").cv_MAE.to_dict())
        print("KNN k (train CV MAE):", k_cv.round(0).set_index("k").cv_MAE.to_dict())
        b = int(stage.cv_MAE.idxmin())
        print("GBM staged: best CV MAE %.0f at %d trees; at 200 trees CV %.0f / train %.0f; at 400 trees CV %.0f / train %.0f"
              % (stage.cv_MAE[b], stage.n_trees[b], stage.cv_MAE[199], stage.train_MAE[199], stage.cv_MAE[399], stage.train_MAE[399]))
    return out


if __name__ == "__main__":
    run()

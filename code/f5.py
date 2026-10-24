"""
AI(인공지능) x 비즈니스 #1 인슈어테크_03_회귀모델 · 정식 코드 f5.py
5편 「최고 모델을 흉내 내는 나무」
무엇을, 어떤 순서로:
 ① 1부 나누기, 최종 모델(GBM)을 학습 데이터로 학습
 ② 대리모델: 회귀나무가 "GBM 의 학습 데이터 예측"을 정답 삼아 배운다 (잎 수 2~16)
 ③ 충실도 = 시험 268명에서 GBM 예측과의 R² · MAE, 정확도 = 실제값과의 MAE · 고액 MAE
 ④ 같은 잎 수로 실제 charges 에 직접 배운 나무와 비교
 ⑤ 잎 수 결정은 학습 데이터 5-Fold 충실도(R²)로만: 충실도 0.95 이상인 가장 작은 나무
 ⑥ 최종 대리모델을 data/surrogate_tree.json 으로 저장 (feature · threshold · left · right · value · n)
"""
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold
import common as C

LEAVES = list(range(2, 17))
FIDELITY_GOAL = 0.95


def r2(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)


def tree_json(t, cols):
    tr = t.tree_
    nodes = []
    for i in range(tr.node_count):
        leaf = tr.children_left[i] == -1
        nodes.append(dict(id=i, feature=None if leaf else cols[tr.feature[i]],
                          threshold=None if leaf else round(float(tr.threshold[i]), 4),
                          left=None if leaf else int(tr.children_left[i]), right=None if leaf else int(tr.children_right[i]),
                          value=round(float(tr.value[i][0][0]), 2), n=int(tr.n_node_samples[i])))
    return nodes


def rules(nodes):
    """잎마다 조건 목록."""
    out = []

    def walk(i, conds):
        n = nodes[i]
        if n["left"] is None:
            out.append(dict(leaf=i, conds=conds, value=n["value"], n=n["n"]))
            return
        f, t = n["feature"], n["threshold"]
        walk(n["left"], conds + ["%s <= %s" % (f, t)])
        walk(n["right"], conds + ["%s > %s" % (f, t)])
    walk(0, [])
    return out


def run(verbose=True):
    df, X, y, info = C.load_clean_xy()
    Xtr, Xte, ytr, yte = C.split(X, y)
    gbm = C.final_model().fit(Xtr, ytr)
    g_tr, g_te = gbm.predict(Xtr), gbm.predict(Xte)
    top = yte.values >= np.quantile(yte, 0.9)
    cv = list(StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(Xtr, C.strata(ytr)))

    rows = []
    for L in LEAVES:
        s = DecisionTreeRegressor(max_leaf_nodes=L, random_state=C.SEED).fit(Xtr, g_tr)
        d = DecisionTreeRegressor(max_leaf_nodes=L, random_state=C.SEED).fit(Xtr, ytr)
        cvf = np.zeros(len(ytr))
        for a, b in cv:
            cvf[b] = DecisionTreeRegressor(max_leaf_nodes=L, random_state=C.SEED).fit(Xtr.iloc[a], g_tr[a]).predict(Xtr.iloc[b])
        ps, pd_ = s.predict(Xte), d.predict(Xte)
        rows.append(dict(leaves=L, depth=s.get_depth(), cv_fid_R2=r2(g_tr, cvf),
                         fid_R2=r2(g_te, ps), fid_MAE=float(np.mean(np.abs(g_te - ps))),
                         MAE=float(np.mean(np.abs(yte - ps))), MAE_top10=float(np.mean(np.abs(yte.values[top] - ps[top]))),
                         direct_MAE=float(np.mean(np.abs(yte - pd_))), direct_MAE_top10=float(np.mean(np.abs(yte.values[top] - pd_[top]))),
                         under_rate=float(np.mean(ps < yte))))
    tab = pd.DataFrame(rows).set_index("leaves")
    ok = tab[tab.cv_fid_R2 >= FIDELITY_GOAL]
    L_final = int(ok.index.min()) if len(ok) else LEAVES[-1]

    sur = DecisionTreeRegressor(max_leaf_nodes=L_final, random_state=C.SEED).fit(Xtr, g_tr)
    cols = list(X.columns)
    nodes = tree_json(sur, cols)
    used = sorted({n["feature"] for n in nodes if n["feature"]})
    js = dict(model="surrogate of GradientBoostingRegressor(%s)" % C.FINAL_PARAMS, leaves=L_final,
              columns=cols, features_used=used, nodes=nodes,
              note="trained on train-set GBM predictions (1,069 rows, split.csv). value = quote in USD.")
    with open("data/surrogate_tree.json", "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False, indent=1)
    gm = C.metrics(yte, g_te)
    out = dict(tab=tab, L_final=L_final, nodes=nodes, rules=rules(nodes), used=used, gbm_metrics=gm,
               g_te=g_te, g_tr=g_tr, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, sur=sur, X=X)
    if verbose:
        print("GBM (teacher) test: MAE=%.0f  MAE_top10=%.0f  R2=%.4f" % (gm["MAE"], gm["MAE_top10"], gm["R2"]))
        print("surrogate = tree trained on GBM train predictions; direct = same-size tree trained on charges")
        print(tab.round({"cv_fid_R2": 4, "fid_R2": 4, "fid_MAE": 0, "MAE": 0, "MAE_top10": 0, "direct_MAE": 0,
                         "direct_MAE_top10": 0, "under_rate": 3}).to_string())
        print()
        print("chosen leaves (smallest with train 5-fold fidelity R2 >= %.2f): %d" % (FIDELITY_GOAL, L_final))
        print("features used:", used)
        print("root split:", nodes[0]["feature"], nodes[0]["threshold"])
        for r in rules(nodes):
            print("  leaf %2d  n=%4d  quote=%9.2f  | %s" % (r["leaf"], r["n"], r["value"], " & ".join(r["conds"])))
    return out


if __name__ == "__main__":
    run()

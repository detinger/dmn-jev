import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from jevdmn.schema import load_schema
from jevdmn.features import typed_frame, signal_frame
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
schema = load_schema("configs/cfpb.yaml")
rows = [json.loads(l) for l in open("data/interim/responses_jev_pilot.jsonl")]
resp = {r["case_id"]: r["response"] for r in rows}
s = pd.read_parquet("data/interim/cfpb_sample.parquet"); s["cid"] = s.complaint_id.astype(str); s = s.set_index("cid").loc[list(resp)]
assert set(s.split_relief.dropna()) <= {"pool"} and set(s.split_routing.dropna()) <= {"pool"}, "non-pool case in pilot!"
print("all pilot cases are pool:", True, "| n =", len(s), "| in relief", s.in_relief.sum(), "| in routing", s.in_routing.sum())
lat = pd.Series([r["latency_ms"] for r in rows if not r["cached"]]); tin = pd.Series([r["usage"]["input_tokens"] for r in rows])
print(f"latency median {lat.median():.0f} ms, p95 {lat.quantile(.95):.0f}, max {lat.max():.0f}; input tokens median {tin.median():.0f}, p95 {tin.quantile(.95):.0f}, max {tin.max()}; output median {np.median([r['usage']['output_tokens'] for r in rows]):.0f}")
t = typed_frame(resp, schema); sig = signal_frame(resp, schema)

print("\n== noul questions: P(true), mean p, share extreme (<.05 or >.95), share uncertain (.3-.7)")
out = []
for q in schema.questions:
    if q.type != "noul": continue
    p = sig[f"{q.id}__p"]
    out.append((q.id, t[q.id].mean(), p.mean(), ((p < .05) | (p > .95)).mean(), ((p > .3) & (p < .7)).mean(), p.nunique()))
print(pd.DataFrame(out, columns=["q", "share_true", "mean_p", "extreme", "uncertain", "n_unique"]).round(3).to_string(index=False))

print("\n== choice/score questions: distribution of typed answers, confidence")
for q in schema.questions:
    if q.type == "noul": continue
    vc = t[q.id].value_counts(normalize=True).round(3)
    dom = [str(x) for x in (q.domain if q.type == "choice" else range(4))]
    never = [d for d in dom if d not in set(map(str, vc.index))]
    print(f"{q.id}: conf median {sig[q.id+'__conf'].median():.2f}, margin median {sig[q.id+'__margin'].median():.2f}; never chosen: {never}")
    print("   ", vc.to_dict())

print("\n== first-option position: share where chosen option is the first listed (choice)")
for q in schema.questions:
    if q.type == "choice":
        first = str(q.domain[0]); print(f"  {q.id}: first option '{first}' chosen {(t[q.id].astype(str)==first).mean():.3f}")

print("\n== product (Jev) vs CFPB product_group")
print(pd.crosstab(s.product_group, t["product"]).to_string())

rt = s[s.in_routing.astype(bool)]
print("\n== routing: Jev holistic queue vs CFPB queue (pool, n=%d), agreement %.3f" % (len(rt), (t.loc[rt.index, "queue_holistic"] == rt.queue).mean()))
print(pd.crosstab(rt.queue, t.loc[rt.index, "queue_holistic"], margins=True).to_string())

rl = s[s.in_relief.astype(bool)]
y = (rl.relief == "relief").astype(int); p = sig.loc[rl.index, "relief_holistic__p"]
from sklearn.metrics import roc_auc_score
print("\n== relief (pool, n=%d): base rate %.3f, Jev says relief %.3f, AUC %.3f, acc@0.5 %.3f" % (len(rl), y.mean(), (p >= .5).mean(), roc_auc_score(y, p), ((p >= .5) == y).mean()))
print("   p histogram:", np.histogram(p, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001])[0].tolist())

print("\n== fact questions vs relief label (pool): mean typed answer by label")
facts = [q.id for q in schema.facts if q.type == "noul"]
print(t.loc[rl.index, facts].astype(float).groupby(y.values).mean().T.round(2).to_string())

print("\n== correlation between noul facts (|r|>0.6)")
c = sig[[f + "__p" for f in facts]].corr()
for i, a in enumerate(c.index):
    for b in c.columns[i + 1:]:
        if abs(c.loc[a, b]) > .6: print(f"  {a} ~ {b}: {c.loc[a, b]:.2f}")

print("\n== harm expected value vs argmax; harm by label")
ev = pd.Series({cid: r["answers"]["harm"]["score"] for cid, r in resp.items()})
print("  mean |EV - argmax|:", round((ev - t["harm"]).abs().mean(), 3))
print(pd.crosstab(rl.relief, t.loc[rl.index, "harm"], normalize="index").round(2).to_string())

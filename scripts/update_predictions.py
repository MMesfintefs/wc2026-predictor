"""
WC2026 Predictor — trains Elo + ML model, pulls latest results,
predicts remaining matches, resolves bracket, writes data/predictions.json
Runs inside GitHub Actions. 100% free data sources.
"""
import json, math, os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ── CONFIG ──────────────────────────────────────────
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
TRAIN_FROM = "2016-01-01"
WC_START   = "2026-06-10"
OUT_PATH   = "data/predictions.json"
MANUAL_PATH = "data/manual_results.json"

GROUPS = {
 "A": ["Mexico","South Africa","South Korea","Czech Republic"],
 "B": ["Canada","Bosnia and Herzegovina","Qatar","Switzerland"],
 "C": ["Brazil","Morocco","Haiti","Scotland"],
 "D": ["United States","Paraguay","Australia","Turkey"],
 "E": ["Germany","Curaçao","Ivory Coast","Ecuador"],
 "F": ["Netherlands","Japan","Sweden","Tunisia"],
 "G": ["Belgium","Egypt","Iran","New Zealand"],
 "H": ["Spain","Cape Verde","Saudi Arabia","Uruguay"],
 "I": ["France","Senegal","Iraq","Norway"],
 "J": ["Argentina","Algeria","Austria","Jordan"],
 "K": ["Portugal","DR Congo","Uzbekistan","Colombia"],
 "L": ["England","Croatia","Ghana","Panama"],
}
# dataset name → our name (extend if diagnostics flag unknowns)
NAME_FIX = {"Türkiye":"Turkey","Korea Republic":"South Korea","Czechia":"Czech Republic",
 "Côte d'Ivoire":"Ivory Coast","IR Iran":"Iran","Cabo Verde":"Cape Verde",
 "Congo DR":"DR Congo","USA":"United States","Curacao":"Curaçao"}

# Official R32 schedule: (match_id, home_slot, away_slot, date)
R32 = [(73,"2A","2B","Jun 28"),(74,"1C","2F","Jun 29"),(75,"1E","3ABCDF","Jun 29"),
 (76,"1F","2C","Jun 29"),(77,"2E","2I","Jun 30"),(78,"1I","3CDFGH","Jun 30"),
 (79,"1A","3CEFHI","Jun 30"),(80,"1L","3EHIJK","Jul 1"),(81,"1G","3AEHIJ","Jul 1"),
 (82,"1D","3BEFIJ","Jul 1"),(83,"1H","2J","Jul 2"),(84,"2K","2L","Jul 2"),
 (85,"1B","3EFGIJ","Jul 2"),(86,"2D","2G","Jul 3"),(87,"1J","2H","Jul 3"),
 (88,"1K","3DEIJL","Jul 3")]
R16 = [(89,73,75,"Jul 4"),(90,74,77,"Jul 4"),(91,76,78,"Jul 5"),(92,79,80,"Jul 5"),
 (93,83,84,"Jul 6"),(94,81,82,"Jul 6"),(95,86,88,"Jul 7"),(96,85,87,"Jul 7")]
QF  = [(97,89,90,"Jul 9"),(98,93,94,"Jul 10"),(99,91,92,"Jul 11"),(100,95,96,"Jul 11")]
SF  = [(101,97,98,"Jul 14"),(102,99,100,"Jul 15")]
FINAL = (104,101,102,"Jul 19")

# ── LOAD DATA ───────────────────────────────────────
def load():
    df = pd.read_csv(RESULTS_URL)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score","away_score"])
    for c in ("home_team","away_team"):
        df[c] = df[c].replace(NAME_FIX)
    df = df[df["date"] >= TRAIN_FROM].sort_values("date").reset_index(drop=True)
    # merge manual results (things martj42 hasn't added yet)
    if os.path.exists(MANUAL_PATH):
        man = json.load(open(MANUAL_PATH))
        rows = []
        for m in man.get("results", []):
            key = (m["home"], m["away"], str(m["date"]))
            exists = ((df.home_team==m["home"]) & (df.away_team==m["away"]) &
                      (df.date==pd.to_datetime(m["date"]))).any()
            if not exists:
                rows.append({"date":pd.to_datetime(m["date"]),"home_team":m["home"],
                 "away_team":m["away"],"home_score":m["hs"],"away_score":m["as"],
                 "tournament":"FIFA World Cup","city":"","country":"","neutral":True})
        if rows:
            df = pd.concat([df,pd.DataFrame(rows)]).sort_values("date").reset_index(drop=True)
    return df

# ── ELO ENGINE ──────────────────────────────────────
def k_factor(t):
    t = str(t)
    if "World Cup" in t and "qualification" not in t: return 60
    if "qualification" in t: return 40
    for big in ("Euro","Copa Am","African Cup","Gold Cup","Asian Cup"):
        if big in t and "qualification" not in t: return 50
    if "Nations League" in t: return 40
    if "Friendly" in t: return 20
    return 30

def margin_mult(gd):
    gd = abs(gd)
    return 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8

def expected(ra, rb): return 1/(1+10**((rb-ra)/400))

# ── BUILD RATINGS + TRAINING FEATURES ───────────────
def build(df):
    elo, hist = {}, {}
    X, y, w, meta = [], [], [], []
    for r in df.itertuples():
        h, a = r.home_team, r.away_team
        elo.setdefault(h,1500); elo.setdefault(a,1500)
        hist.setdefault(h,[]); hist.setdefault(a,[])
        adv = 0 if r.neutral else 60
        eh = expected(elo[h]+adv, elo[a])
        res = 1.0 if r.home_score>r.away_score else 0.0 if r.home_score<r.away_score else 0.5

        def feats(t, opp_elo_now):
            hh = hist[t][-10:]
            if len(hh) < 8: return None
            perf = np.mean([x["pts"]-x["exp"] for x in hh])
            gf   = np.mean([x["gf"] for x in hh]); ga = np.mean([x["ga"] for x in hh])
            oe   = np.mean([x["opp"] for x in hh])
            rest = min((r.date - hh[-1]["date"]).days, 30)
            return perf, gf, ga, oe, rest

        fh, fa = feats(h, elo[a]), feats(a, elo[h])
        if fh and fa:
            row = [elo[h]+adv-elo[a], fh[0]-fa[0], fh[1]-fa[1], fh[2]-fa[2],
                   fh[3]-fa[3], fh[4]-fa[4]]
            X.append(row); y.append(2 if res==1 else 1 if res==0.5 else 0)
            yrs = (r.date - pd.Timestamp(TRAIN_FROM)).days/365
            w.append((2.0 if k_factor(r.tournament)==60 else 1.0) * math.exp(0.15*yrs))
            meta.append(r.date)
        # update state AFTER extracting pre-match features
        k = k_factor(r.tournament) * margin_mult(r.home_score - r.away_score)
        elo[h] += k*(res-eh); elo[a] += k*((1-res)-(1-eh))
        hist[h].append({"date":r.date,"pts":res,"exp":eh,"gf":r.home_score,"ga":r.away_score,"opp":elo[a]})
        hist[a].append({"date":r.date,"pts":1-res,"exp":1-eh,"gf":r.away_score,"ga":r.home_score,"opp":elo[h]})
    return elo, hist, np.array(X), np.array(y), np.array(w)

def team_feats(t, opp, elo, hist, today):
    hh = hist.get(t,[])[-10:]
    if len(hh) < 5: return [0,0,1.3,1.3,1500,7]
    perf = np.mean([x["pts"]-x["exp"] for x in hh])
    gf = np.mean([x["gf"] for x in hh]); ga = np.mean([x["ga"] for x in hh])
    oe = np.mean([x["opp"] for x in hh])
    rest = min((today - hh[-1]["date"]).days, 30)
    return [perf, gf, ga, oe, rest]

def predict(model, a, b, elo, hist, today, knockout=False):
    fa, fb = team_feats(a,b,elo,hist,today), team_feats(b,a,elo,hist,today)
    x = [[elo.get(a,1500)-elo.get(b,1500), fa[0]-fb[0], fa[1]-fb[1],
          fa[2]-fb[2], fa[3]-fb[3], fa[4]-fb[4]]]
    pL, pD, pW = model.predict_proba(x)[0]   # classes 0,1,2 = away,draw,home
    if knockout:
        tot = pW + pL
        pW, pL, pD = pW + pD*(pW/tot), pL + pD*(pL/tot), 0.0
    return round(pW*100,1), round(pD*100,1), round(pL*100,1)

# ── MAIN ────────────────────────────────────────────
def main():
    df = load()
    elo, hist, X, y, w = build(df)
    model = LogisticRegression(max_iter=2000, C=0.5)
    model.fit(X, y, sample_weight=w)
    today = pd.Timestamp.now()

    # 2026 WC results so far
    wc = df[(df.tournament.str.contains("World Cup")) &
            (~df.tournament.str.contains("qualification")) &
            (df.date >= WC_START)]
    all_teams = {t:g for g,ts in GROUPS.items() for t in ts}
    results = []
    for r in wc.itertuples():
        if r.home_team in all_teams and r.away_team in all_teams:
            results.append({"g":all_teams[r.home_team],"home":r.home_team,"away":r.away_team,
              "hs":int(r.home_score),"as":int(r.away_score),"date":str(r.date.date())})

    # standings
    standings = {}
    for g, teams in GROUPS.items():
        s = {t:{"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0,"pts":0} for t in teams}
        for m in results:
            if m["g"]!=g: continue
            h,a,hs,as_ = m["home"],m["away"],m["hs"],m["as"]
            s[h]["mp"]+=1; s[a]["mp"]+=1
            s[h]["gf"]+=hs; s[h]["ga"]+=as_; s[a]["gf"]+=as_; s[a]["ga"]+=hs
            if hs>as_: s[h]["w"]+=1; s[h]["pts"]+=3; s[a]["l"]+=1
            elif hs<as_: s[a]["w"]+=1; s[a]["pts"]+=3; s[h]["l"]+=1
            else: s[h]["d"]+=1; s[a]["d"]+=1; s[h]["pts"]+=1; s[a]["pts"]+=1
        order = sorted(teams, key=lambda t:(s[t]["pts"], s[t]["gf"]-s[t]["ga"],
                 s[t]["gf"], elo.get(t,1500)), reverse=True)
        standings[g] = [{"team":t, **s[t], "gd":s[t]["gf"]-s[t]["ga"],
                         "elo":round(elo.get(t,1500))} for t in order]

    # resolve bracket slots
    def slot(code):
        if code[0] in "12":
            return standings[code[1]][int(code[0])-1]["team"]
        # 3rd-place: pick best-ranked available third among allowed groups (greedy approx of FIFA table)
        allowed = list(code[1:])
        thirds = sorted([standings[g][2] for g in allowed],
          key=lambda r:(r["pts"],r["gd"],r["gf"],r["elo"]), reverse=True)
        for t in thirds:
            if t["team"] not in slot.used:
                slot.used.add(t["team"]); return t["team"]
        return thirds[0]["team"]
    slot.used = set()

    def run_round(pairs, prev):
        out = []
        for mid, aa, bb, date in pairs:
            A = slot(aa) if isinstance(aa,str) else prev[aa]
            B = slot(bb) if isinstance(bb,str) else prev[bb]
            pW,pD,pL = predict(model, A, B, elo, hist, today, knockout=True)
            out.append({"id":mid,"date":date,"home":A,"away":B,
                        "pHome":pW,"pAway":pL,"winner":A if pW>=pL else B})
        return out

    r32 = run_round(R32, {})
    prev = {m["id"]:m["winner"] for m in r32}
    r16 = run_round(R16, prev); prev.update({m["id"]:m["winner"] for m in r16})
    qf  = run_round(QF, prev);  prev.update({m["id"]:m["winner"] for m in qf})
    sf  = run_round(SF, prev);  prev.update({m["id"]:m["winner"] for m in sf})
    fin = run_round([FINAL], prev)[0]

    # Monte Carlo champion probabilities (simulate knockout 5000x from current bracket)
    champs = {}
    for _ in range(5000):
        cur = {}
        for mid,aa,bb,_ in R32:
            A,B = slot(aa) if isinstance(aa,str) else cur[aa], None
        # simpler: reuse deterministic slots, sample winners
        winners = {}
        def sim(pairs, src):
            for mid,aa,bb,_ in pairs:
                A = slot(aa) if isinstance(aa,str) else winners[aa]
                B = slot(bb) if isinstance(bb,str) else winners[bb]
                pW,_,pL = predict(model,A,B,elo,hist,today,knockout=True)
                winners[mid] = A if np.random.rand() < pW/(pW+pL) else B
        slot.used = set()
        sim(R32,{}); sim(R16,winners); sim(QF,winners); sim(SF,winners); sim([FINAL],winners)
        c = winners[104]; champs[c] = champs.get(c,0)+1
    champ_probs = {k: round(v/5000*100,1) for k,v in
                   sorted(champs.items(), key=lambda x:-x[1])}

    # backtest on played WC matches
    correct, n = 0, 0
    for m in results:
        pW,pD,pL = predict(model, m["home"], m["away"], elo, hist, today)
        pick = "H" if pW>=max(pD,pL) else "D" if pD>=pL else "A"
        act  = "H" if m["hs"]>m["as"] else "A" if m["hs"]<m["as"] else "D"
        correct += (pick==act); n += 1

    out = {
      "generated_at": datetime.now(timezone.utc).isoformat(),
      "model": "Elo(margin+importance) + multinomial LR (form, opp-adj, rest) | trained "+TRAIN_FROM+"→",
      "backtest_group_accuracy": round(correct/max(n,1)*100,1),
      "matches_played": n,
      "elo": {t: round(elo.get(t,1500)) for t in all_teams},
      "results": results,
      "standings": standings,
      "bracket": {"r32":r32,"r16":r16,"qf":qf,"sf":sf,"final":fin},
      "champion_probs": champ_probs,
    }
    os.makedirs("data", exist_ok=True)
    json.dump(out, open(OUT_PATH,"w"), indent=1)
    print(f"OK — {n} results | backtest {out['backtest_group_accuracy']}% | top: {list(champ_probs)[:3]}")

if __name__ == "__main__":
    main()

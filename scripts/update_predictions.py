"""
WC2026 Predictor v2 — honest edition.
Fix 1: train pre-2026 only, holdout = 2026 WC group stage (real out-of-sample metrics)
Fix 2: compares LogisticRegression vs RandomForest vs GradientBoosting vs Poisson
Fix 3: optional player injury/suspension adjustment (Kaggle swaptr dataset, via secrets)
Fix 4: opponent-quality features incl. performance vs top opponents specifically
Fix 5: knockout bracket hardcoded from ESPN/FIFA actual results (no slot re-derivation)
Fix 7: optional StatsBomb xG features (data/statsbomb_features.csv if present)
"""
import json, math, os, subprocess
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from scipy.stats import poisson as pois

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
TRAIN_FROM, TRAIN_UNTIL = "2016-01-01", "2026-06-01"
WC_GROUP_START, WC_GROUP_END = "2026-06-10", "2026-06-27"
OUT = "data/predictions.json"

NAME_FIX = {"Türkiye":"Turkey","Korea Republic":"South Korea","Czechia":"Czech Republic",
 "Côte d'Ivoire":"Ivory Coast","IR Iran":"Iran","Cabo Verde":"Cape Verde",
 "Congo DR":"DR Congo","USA":"United States","Curacao":"Curaçao","Bosnia-Herz":"Bosnia and Herzegovina"}

GROUPS = {"A":["Mexico","South Africa","South Korea","Czech Republic"],
 "B":["Canada","Bosnia and Herzegovina","Qatar","Switzerland"],
 "C":["Brazil","Morocco","Haiti","Scotland"],"D":["United States","Paraguay","Australia","Turkey"],
 "E":["Germany","Curaçao","Ivory Coast","Ecuador"],"F":["Netherlands","Japan","Sweden","Tunisia"],
 "G":["Belgium","Egypt","Iran","New Zealand"],"H":["Spain","Cape Verde","Saudi Arabia","Uruguay"],
 "I":["France","Senegal","Iraq","Norway"],"J":["Argentina","Algeria","Austria","Jordan"],
 "K":["Portugal","DR Congo","Uzbekistan","Colombia"],"L":["England","Croatia","Ghana","Panama"]}
ALL_TEAMS = {t:g for g,ts in GROUPS.items() for t in ts}

# ══ FIX 5: ACTUAL bracket from ESPN (edit scores here / or via manual_results.json "knockout") ══
R32 = [
 {"id":73,"home":"South Africa","away":"Canada","score":[0,1],"status":"FT","date":"Jun 28"},
 {"id":74,"home":"Germany","away":"Paraguay","score":[1,1],"pens":[3,4],"status":"PENS","date":"Jun 29"},
 {"id":75,"home":"Netherlands","away":"Morocco","score":[1,1],"pens":[2,3],"status":"PENS","date":"Jun 29"},
 {"id":76,"home":"Brazil","away":"Japan","score":[2,1],"status":"FT","date":"Jun 29"},
 {"id":77,"home":"France","away":"Sweden","score":[3,0],"status":"FT","date":"Jun 30"},
 {"id":78,"home":"Ivory Coast","away":"Norway","score":[1,2],"status":"FT","date":"Jun 30"},
 {"id":79,"home":"Mexico","away":"Ecuador","score":[2,0],"status":"FT","date":"Jun 30"},
 {"id":80,"home":"England","away":"DR Congo","score":[2,1],"status":"FT","date":"Jul 1"},
 {"id":81,"home":"United States","away":"Bosnia and Herzegovina","score":[2,0],"status":"FT","date":"Jul 1"},
 {"id":82,"home":"Belgium","away":"Senegal","score":[3,2],"status":"AET","date":"Jul 1"},
 {"id":83,"home":"Portugal","away":"Croatia","score":[2,1],"status":"FT","date":"Jul 2"},
 {"id":84,"home":"Spain","away":"Austria","score":[3,0],"status":"FT","date":"Jul 2"},
 {"id":85,"home":"Switzerland","away":"Algeria","score":[2,0],"status":"FT","date":"Jul 2"},
 {"id":86,"home":"Argentina","away":"Cape Verde","status":"SCHEDULED","date":"Jul 3"},
 {"id":87,"home":"Colombia","away":"Ghana","status":"SCHEDULED","date":"Jul 3"},
 {"id":88,"home":"Australia","away":"Egypt","score":[1,1],"pens":[2,4],"status":"PENS","date":"Jul 3"},
]
R16_FEEDS = [(89,73,75,"Jul 4"),(90,74,77,"Jul 4"),(91,76,78,"Jul 5"),(92,79,80,"Jul 5"),
             (93,83,84,"Jul 6"),(94,81,82,"Jul 6"),(95,86,88,"Jul 7"),(96,85,87,"Jul 7")]
QF_FEEDS = [(97,89,90,"Jul 9"),(98,93,94,"Jul 10"),(99,91,92,"Jul 11"),(100,95,96,"Jul 11")]
SF_FEEDS = [(101,97,98,"Jul 14"),(102,99,100,"Jul 15")]
FN_FEED  = (104,101,102,"Jul 19")

def winner_of(m):
    if m.get("status") in (None,"SCHEDULED","LIVE"): return None
    if m.get("pens"): return m["home"] if m["pens"][0]>m["pens"][1] else m["away"]
    if m.get("score"):
        if m["score"][0]>m["score"][1]: return m["home"]
        if m["score"][0]<m["score"][1]: return m["away"]
    return None

# ══ Load results ══
def load():
    df = pd.read_csv(RESULTS_URL)
    df["date"]=pd.to_datetime(df["date"]); df=df.dropna(subset=["home_score","away_score"])
    for c in("home_team","away_team"): df[c]=df[c].replace(NAME_FIX)
    df=df[df["date"]>=TRAIN_FROM].sort_values("date").reset_index(drop=True)
    man_path="data/manual_results.json"
    ko_over=[]
    if os.path.exists(man_path):
        man=json.load(open(man_path)); rows=[]
        for m in man.get("results",[]):
            if not ((df.home_team==m["home"])&(df.away_team==m["away"])&(df.date==pd.to_datetime(m["date"]))).any():
                rows.append({"date":pd.to_datetime(m["date"]),"home_team":m["home"],"away_team":m["away"],
                 "home_score":m["hs"],"away_score":m["as"],"tournament":"FIFA World Cup","neutral":True})
        if rows: df=pd.concat([df,pd.DataFrame(rows)]).sort_values("date").reset_index(drop=True)
        ko_over=man.get("knockout",[])
    for o in ko_over:                      # override knockout scores from manual file
        for m in R32:
            if m["id"]==o["id"]: m.update(o)
    return df

# ══ StatsBomb xG features (Fix 7, optional file) ══
def load_xg():
    p="data/statsbomb_features.csv"
    if not os.path.exists(p): return None
    xf=pd.read_csv(p)   # cols: team,season_end,xg90,xga90
    xf["season_end"]=pd.to_datetime(xf["season_end"])
    return xf
def xg_at(xf,team,date):
    if xf is None: return 1.3,1.3
    r=xf[(xf.team==team)&(xf.season_end<date)].sort_values("season_end")
    if len(r)==0: return 1.3,1.3
    return float(r.iloc[-1].xg90), float(r.iloc[-1].xga90)

# ══ Elo + rolling features ══
def k_factor(t):
    t=str(t)
    if "World Cup" in t and "qualification" not in t: return 60
    if "qualification" in t: return 40
    for b in("Euro","Copa Am","African Cup","Gold Cup","Asian Cup"):
        if b in t and "qualification" not in t: return 50
    if "Nations League" in t: return 40
    if "Friendly" in t: return 20
    return 30
def mmult(gd): gd=abs(gd); return 1.0 if gd<=1 else 1.5 if gd==2 else (11+gd)/8
def exp_(ra,rb): return 1/(1+10**((rb-ra)/400))
TOP_ELO=1700   # "big team" threshold (Fix 4)

def build(df,xf):
    elo,hist={},{}
    X,y,dates=[],[],[]
    XP,yP=[],[]                     # Poisson long-format
    for r in df.itertuples():
        h,a=r.home_team,r.away_team
        elo.setdefault(h,1500);elo.setdefault(a,1500);hist.setdefault(h,[]);hist.setdefault(a,[])
        adv=0 if r.neutral else 60
        eh=exp_(elo[h]+adv,elo[a])
        res=1.0 if r.home_score>r.away_score else 0.0 if r.home_score<r.away_score else 0.5
        def F(t):
            hh=hist[t][-10:]
            if len(hh)<8: return None
            perf=np.mean([x["pts"]-x["exp"] for x in hh])
            big=[x["pts"]-x["exp"] for x in hh if x["opp"]>=TOP_ELO]
            perf_big=np.mean(big) if big else 0.0        # Fix 4: perf vs top opponents only
            gf=np.mean([x["gf"] for x in hh]); ga=np.mean([x["ga"] for x in hh])
            oe=np.mean([x["opp"] for x in hh])
            rest=min((r.date-hh[-1]["date"]).days,30)
            return perf,perf_big,gf,ga,oe,rest
        fh,fa=F(h),F(a)
        if fh and fa:
            xgh,xgah=xg_at(xf,h,r.date); xga,xgaa=xg_at(xf,a,r.date)
            row=[elo[h]+adv-elo[a],fh[0]-fa[0],fh[1]-fa[1],fh[2]-fa[2],fh[3]-fa[3],
                 fh[4]-fa[4],fh[5]-fa[5],xgh-xga,xgah-xgaa]
            X.append(row); y.append(2 if res==1 else 1 if res==0.5 else 0); dates.append(r.date)
            XP.append([elo[h]+adv-elo[a],fh[2],fa[3],xgh,xgaa]); yP.append(r.home_score)
            XP.append([elo[a]-elo[h]-adv,fa[2],fh[3],xga,xgah]); yP.append(r.away_score)
        k=k_factor(r.tournament)*mmult(r.home_score-r.away_score)
        elo[h]+=k*(res-eh); elo[a]+=k*((1-res)-(1-eh))
        hist[h].append({"date":r.date,"pts":res,"exp":eh,"gf":r.home_score,"ga":r.away_score,"opp":elo[a]})
        hist[a].append({"date":r.date,"pts":1-res,"exp":1-eh,"gf":r.away_score,"ga":r.home_score,"opp":elo[h]})
    return elo,hist,np.array(X),np.array(y),pd.Series(dates),np.array(XP),np.array(yP)

FEATS=["elo_diff","perf_vs_exp_diff","perf_vs_TOP20_diff","goals_for_diff","goals_against_diff",
       "opp_strength_faced_diff","rest_days_diff","xG90_diff(StatsBomb)","xGA90_diff(StatsBomb)"]

def team_row(a,b,elo,hist,xf,today):
    def F(t):
        hh=hist.get(t,[])[-10:]
        if len(hh)<5: return 0,0,1.3,1.3,1500,7
        perf=np.mean([x["pts"]-x["exp"] for x in hh])
        big=[x["pts"]-x["exp"] for x in hh if x["opp"]>=TOP_ELO]
        return (perf,(np.mean(big) if big else 0.0),np.mean([x["gf"] for x in hh]),
                np.mean([x["ga"] for x in hh]),np.mean([x["opp"] for x in hh]),
                min((today-hh[-1]["date"]).days,30))
    fa,fb=F(a),F(b)
    xga_,xgaa_=xg_at(xf,a,today); xgb_,xgab_=xg_at(xf,b,today)
    return [elo.get(a,1500)-elo.get(b,1500),fa[0]-fb[0],fa[1]-fb[1],fa[2]-fb[2],
            fa[3]-fb[3],fa[4]-fb[4],fa[5]-fb[5],xga_-xgb_,xgaa_-xgab_]

# ══ Fix 3: player availability adjustment (heuristic; only if Kaggle secrets set) ══
def player_adjustments():
    if not(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        return {}, "skipped — no Kaggle credentials configured"
    try:
        subprocess.run(["kaggle","datasets","download","swaptr/fifa-wc-2026-players",
                        "-p","data/players","--unzip","-q"],check=True,timeout=300)
        pen={}
        for f in os.listdir("data/players"):
            if not f.endswith(".csv"): continue
            d=pd.read_csv(f"data/players/{f}")
            cols={c.lower():c for c in d.columns}
            team_c=next((cols[k] for k in("team","country","nationality","nation") if k in cols),None)
            if not team_c: continue
            inj_c=next((cols[k] for k in cols if "injur" in k or "status" in k),None)
            red_c=next((cols[k] for k in cols if "red" in k),None)
            for _,row in d.iterrows():
                t=NAME_FIX.get(str(row[team_c]),str(row[team_c]))
                if t not in ALL_TEAMS: continue
                p=0
                if inj_c and isinstance(row.get(inj_c),str) and any(w in row[inj_c].lower() for w in("injur","out","doubt")): p+=15
                if red_c and pd.notna(row.get(red_c)) and float(row.get(red_c) or 0)>=1: p+=10
                if p: pen[t]=min(pen.get(t,0)+p,60)
        return pen, f"applied — {len(pen)} teams have Elo penalties (−15/injury, −10/red-card susp., cap −60). Heuristic: no historical injury data exists on the free tier to train a coefficient."
    except Exception as e:
        return {}, f"failed ({type(e).__name__}) — proceeding without"

# ══ MAIN ══
def main():
    df=load(); xf=load_xg()
    elo,hist,X,y,dts,XP,yP=build(df,xf)

    # Fix 1: honest split
    tr = dts < TRAIN_UNTIL
    wc = df[(df.date>=WC_GROUP_START)&(df.date<=WC_GROUP_END)&
            (df.home_team.isin(ALL_TEAMS))&(df.away_team.isin(ALL_TEAMS))]
    te = (dts>=WC_GROUP_START)&(dts<=WC_GROUP_END)
    def brier(p,yt): 
        oh=np.zeros_like(p); oh[np.arange(len(yt)),yt]=1; return float(np.mean(np.sum((p-oh)**2,1)))

    # Fix 2: model comparison on identical features
    cands={"LogisticRegression":LogisticRegression(max_iter=2000,C=0.5),
           "RandomForest":RandomForestClassifier(n_estimators=400,min_samples_leaf=25,random_state=0),
           "GradientBoosting":GradientBoostingClassifier(random_state=0)}
    comp=[]; fitted={}
    for name,m in cands.items():
        m.fit(X[tr],y[tr]); p=m.predict_proba(X[te])
        comp.append({"model":name,"holdout_matches":int(te.sum()),
          "accuracy":round(accuracy_score(y[te],p.argmax(1))*100,1),
          "log_loss":round(log_loss(y[te],p,labels=[0,1,2]),4),"brier":round(brier(p,y[te]),4)})
        fitted[name]=m
    # Poisson (goals) candidate
    trP=np.repeat(tr.values,2); teP=np.repeat(te.values,2)
    pr=PoissonRegressor(alpha=0.5,max_iter=1000).fit(XP[trP],yP[trP])
    idx=np.where(te.values)[0]; probs=[]
    for i in idx:
        lh=max(.2,pr.predict(XP[2*i:2*i+1])[0]); la=max(.2,pr.predict(XP[2*i+1:2*i+2])[0])
        M=np.outer([pois.pmf(k,lh) for k in range(9)],[pois.pmf(k,la) for k in range(9)])
        probs.append([np.triu(M,1).sum(),np.trace(M),np.tril(M,-1).sum()])
    pP=np.array(probs)
    comp.append({"model":"PoissonGoals","holdout_matches":int(te.sum()),
      "accuracy":round(accuracy_score(y[te],pP.argmax(1))*100,1),
      "log_loss":round(log_loss(y[te],pP,labels=[0,1,2]),4),"brier":round(brier(pP,y[te]),4)})
    best=min([c for c in comp if c["model"]!="PoissonGoals"],key=lambda c:c["log_loss"])
    # (Poisson kept as reference; classifier used for W/D/L going forward)
    model=fitted[best["model"]]; model.fit(X,y)   # refit on ALL data for forward preds (after honest eval)

    pen,pen_note=player_adjustments()
    eloA={t:elo.get(t,1500)-pen.get(t,0) for t in ALL_TEAMS}
    today=pd.Timestamp.now()
    def P(a,b):
        r=team_row(a,b,eloA,hist,xf,today)
        pl,pd_,pw=model.predict_proba([r])[0]
        tot=pw+pl
        return round((pw+pd_*pw/tot)*100,1), round((pl+pd_*pl/tot)*100,1)

    # group standings (played matches)
    standings={}
    for g,teams in GROUPS.items():
        s={t:{"mp":0,"pts":0,"gf":0,"ga":0} for t in teams}
        for r in wc.itertuples():
            if r.home_team in teams and r.away_team in teams:
                h,a,hs,as_=r.home_team,r.away_team,int(r.home_score),int(r.away_score)
                s[h]["mp"]+=1;s[a]["mp"]+=1;s[h]["gf"]+=hs;s[h]["ga"]+=as_;s[a]["gf"]+=as_;s[a]["ga"]+=hs
                s[h]["pts"]+=3 if hs>as_ else 1 if hs==as_ else 0
                s[a]["pts"]+=3 if as_>hs else 1 if hs==as_ else 0
        order=sorted(teams,key=lambda t:(s[t]["pts"],s[t]["gf"]-s[t]["ga"],s[t]["gf"],eloA[t]),reverse=True)
        standings[g]=[{"team":t,**s[t],"gd":s[t]["gf"]-s[t]["ga"],"elo":round(eloA[t])} for t in order]

    # bracket: actuals + predictions
    def make_round(feeds,pool):
        out=[]
        for mid,f1,f2,date in feeds:
            A,B=pool.get(f1),pool.get(f2)
            m={"id":mid,"date":date,"home":A,"away":B,"status":"SCHEDULED"}
            if A and B:
                m["pHome"],m["pAway"]=P(A,B)
                m["predicted"]=A if m["pHome"]>=m["pAway"] else B
            out.append(m)
        return out
    pool={}
    r32=[]
    for m in R32:
        mm=dict(m); w=winner_of(mm)
        if w: mm["winner"]=w
        else:
            mm["pHome"],mm["pAway"]=P(mm["home"],mm["away"])
            mm["predicted"]=mm["home"] if mm["pHome"]>=mm["pAway"] else mm["away"]
        pool[mm["id"]]=w or mm.get("predicted")
        r32.append(mm)
    r16=make_round(R16_FEEDS,pool); pool.update({m["id"]:m.get("winner") or m.get("predicted") for m in r16})
    qf=make_round(QF_FEEDS,pool);  pool.update({m["id"]:m.get("winner") or m.get("predicted") for m in qf})
    sf=make_round(SF_FEEDS,pool);  pool.update({m["id"]:m.get("winner") or m.get("predicted") for m in sf})
    fin=make_round([FN_FEED],pool)[0]

    # champion Monte Carlo (respects actual results; samples undecided)
    champs={}
    for _ in range(5000):
        w={}
        for m in R32:
            act=winner_of(m)
            if act: w[m["id"]]=act
            else:
                ph,pa=P(m["home"],m["away"]); w[m["id"]]=m["home"] if np.random.rand()<ph/(ph+pa) else m["away"]
        for feeds in (R16_FEEDS,QF_FEEDS,SF_FEEDS,[FN_FEED]):
            for mid,f1,f2,_ in feeds:
                A,B=w[f1],w[f2]; ph,pa=P(A,B)
                w[mid]=A if np.random.rand()<ph/(ph+pa) else B
        champs[w[104]]=champs.get(w[104],0)+1
    champ_probs={k:round(v/50,1) for k,v in sorted(champs.items(),key=lambda x:-x[1])}

    json.dump({
      "generated_at":datetime.now(timezone.utc).isoformat(),
      "methodology":{"candidates_compared":comp,"selected_model":best["model"],
        "selection_rule":"lowest log-loss on 2026 WC group-stage holdout (never seen in training)",
        "features":FEATS,"train_window":f"{TRAIN_FROM} → {TRAIN_UNTIL} (holdout: WC group stage)",
        "player_adjustment":pen_note,
        "statsbomb_xg":"loaded" if xf is not None else "not present — run scripts/build_statsbomb_features.py once and commit the CSV"},
      "standings":standings,
      "bracket":{"r32":r32,"r16":r16,"qf":qf,"sf":sf,"final":fin},
      "champion_probs":champ_probs,
    },open(OUT,"w"),indent=1)
    print("OK | best:",best["model"],"| holdout log-loss:",best["log_loss"],"| acc:",best["accuracy"])

if __name__=="__main__": main()

"""WC2026 Predictor v4 — fixed Poisson xG, expanded model zoo (trees/bagging/XGB/ensemble)."""
import json, math, os, subprocess
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, BaggingClassifier, HistGradientBoostingClassifier, VotingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from scipy.stats import poisson as pois
try:
    from xgboost import XGBClassifier
    HAS_XGB=True
except Exception:
    HAS_XGB=False

RESULTS_URL="https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
TRAIN_FROM,TRAIN_UNTIL="2016-01-01","2026-06-01"
WC_GROUP_START,WC_GROUP_END="2026-06-10","2026-06-27"
OUT="data/predictions.json"
NAME_FIX={"Türkiye":"Turkey","Korea Republic":"South Korea","Czechia":"Czech Republic",
 "Côte d'Ivoire":"Ivory Coast","IR Iran":"Iran","Cabo Verde":"Cape Verde","Congo DR":"DR Congo",
 "USA":"United States","Curacao":"Curaçao","Bosnia-Herz":"Bosnia and Herzegovina"}
GROUPS={"A":["Mexico","South Africa","South Korea","Czech Republic"],
 "B":["Canada","Bosnia and Herzegovina","Qatar","Switzerland"],
 "C":["Brazil","Morocco","Haiti","Scotland"],"D":["United States","Paraguay","Australia","Turkey"],
 "E":["Germany","Curaçao","Ivory Coast","Ecuador"],"F":["Netherlands","Japan","Sweden","Tunisia"],
 "G":["Belgium","Egypt","Iran","New Zealand"],"H":["Spain","Cape Verde","Saudi Arabia","Uruguay"],
 "I":["France","Senegal","Iraq","Norway"],"J":["Argentina","Algeria","Austria","Jordan"],
 "K":["Portugal","DR Congo","Uzbekistan","Colombia"],"L":["England","Croatia","Ghana","Panama"]}
ALL_TEAMS={t:g for g,ts in GROUPS.items() for t in ts}

R32=[{"id":73,"home":"South Africa","away":"Canada","score":[0,1],"status":"FT","date":"Jun 28"},
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
 {"id":88,"home":"Australia","away":"Egypt","score":[1,1],"pens":[2,4],"status":"PENS","date":"Jul 3"}]
R16_FEEDS=[(89,73,75,"Jul 4"),(90,74,77,"Jul 4"),(91,76,78,"Jul 5"),(92,79,80,"Jul 5"),
 (93,83,84,"Jul 6"),(94,81,82,"Jul 6"),(95,86,88,"Jul 7"),(96,85,87,"Jul 7")]
QF_FEEDS=[(97,89,90,"Jul 9"),(98,93,94,"Jul 10"),(99,91,92,"Jul 11"),(100,95,96,"Jul 11")]
SF_FEEDS=[(101,97,98,"Jul 14"),(102,99,100,"Jul 15")]
FN_FEED=(104,101,102,"Jul 19")

def winner_of(m):
    if m.get("status") in (None,"SCHEDULED","LIVE"): return None
    if m.get("pens"): return m["home"] if m["pens"][0]>m["pens"][1] else m["away"]
    if m.get("score"):
        if m["score"][0]>m["score"][1]: return m["home"]
        if m["score"][0]<m["score"][1]: return m["away"]
    return None

def fd_results():
    tok=os.getenv("FOOTBALL_DATA_TOKEN")
    if not tok: return [],"skipped — no FOOTBALL_DATA_TOKEN secret"
    try:
        r=requests.get("https://api.football-data.org/v4/competitions/WC/matches",
            headers={"X-Auth-Token":tok},timeout=30)
        if r.status_code!=200: return [],f"HTTP {r.status_code}"
        out=[]
        for m in r.json().get("matches",[]):
            if m.get("status")!="FINISHED": continue
            ft=m["score"]["fullTime"]
            h=NAME_FIX.get(m["homeTeam"]["name"],m["homeTeam"]["name"])
            a=NAME_FIX.get(m["awayTeam"]["name"],m["awayTeam"]["name"])
            if h in ALL_TEAMS and a in ALL_TEAMS and ft["home"] is not None:
                out.append({"date":m["utcDate"][:10],"home":h,"away":a,"hs":ft["home"],"as":ft["away"]})
        return out,f"filled {len(out)} finished matches"
    except Exception as e: return [],f"failed ({type(e).__name__})"

def load():
    df=pd.read_csv(RESULTS_URL)
    df["date"]=pd.to_datetime(df["date"]); df=df.dropna(subset=["home_score","away_score"])
    for c in("home_team","away_team"): df[c]=df[c].replace(NAME_FIX)
    df=df[df["date"]>=TRAIN_FROM].sort_values("date").reset_index(drop=True)
    extra,fd_note=fd_results()
    man={"results":[],"knockout":[]}
    if os.path.exists("data/manual_results.json"):
        man=json.load(open("data/manual_results.json"))
    rows=[]
    for m in man.get("results",[])+extra:
        if not((df.home_team==m["home"])&(df.away_team==m["away"])&(df.date==pd.to_datetime(m["date"]))).any():
            rows.append({"date":pd.to_datetime(m["date"]),"home_team":m["home"],"away_team":m["away"],
             "home_score":m["hs"],"away_score":m["as"],"tournament":"FIFA World Cup","neutral":True})
    if rows: df=pd.concat([df,pd.DataFrame(rows)]).sort_values("date").reset_index(drop=True)
    for o in man.get("knockout",[]):
        for m in R32:
            if m["id"]==o["id"]: m.update(o)
    return df,fd_note

def load_xg():
    p="data/statsbomb_features.csv"
    if not os.path.exists(p): return None
    xf=pd.read_csv(p); xf["season_end"]=pd.to_datetime(xf["season_end"]); return xf
def xg_at(xf,t,d):
    if xf is None: return 1.3,1.3
    r=xf[(xf.team==t)&(xf.season_end<d)].sort_values("season_end")
    return (1.3,1.3) if len(r)==0 else (float(r.iloc[-1].xg90),float(r.iloc[-1].xga90))

def k_factor(t):
    t=str(t)
    if "World Cup" in t and "qualification" not in t: return 60
    if "qualification" in t: return 40
    for b in("Euro","Copa Am","African Cup","Gold Cup","Asian Cup"):
        if b in t and "qualification" not in t: return 50
    if "Nations League" in t: return 40
    return 20 if "Friendly" in t else 30
def mmult(gd): gd=abs(gd); return 1.0 if gd<=1 else 1.5 if gd==2 else (11+gd)/8
def exp_(ra,rb): return 1/(1+10**((rb-ra)/400))
TOP_ELO=1700

def build(df,xf):
    elo,hist={},{}
    X,y,dates,XP,yP=[],[],[],[],[]
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
            return (perf,(np.mean(big) if big else 0.0),np.mean([x["gf"] for x in hh]),
                    np.mean([x["ga"] for x in hh]),np.mean([x["opp"] for x in hh]),
                    min((r.date-hh[-1]["date"]).days,30))
        fh,fa=F(h),F(a)
        if fh and fa:
            xgh,xgah=xg_at(xf,h,r.date); xga,xgaa=xg_at(xf,a,r.date)
            X.append([elo[h]+adv-elo[a],fh[0]-fa[0],fh[1]-fa[1],fh[2]-fa[2],fh[3]-fa[3],
                      fh[4]-fa[4],fh[5]-fa[5],xgh-xga,xgah-xgaa])
            y.append(2 if res==1 else 1 if res==0.5 else 0); dates.append(r.date)
            XP.append([elo[h]+adv-elo[a],fh[2],fa[3],xgh,xgaa]); yP.append(r.home_score)
            XP.append([elo[a]-elo[h]-adv,fa[2],fh[3],xga,xgah]); yP.append(r.away_score)
        k=k_factor(r.tournament)*mmult(r.home_score-r.away_score)
        elo[h]+=k*(res-eh); elo[a]+=k*((1-res)-(1-eh))
        hist[h].append({"date":r.date,"pts":res,"exp":eh,"gf":r.home_score,"ga":r.away_score,"opp":elo[a]})
        hist[a].append({"date":r.date,"pts":1-res,"exp":1-eh,"gf":r.away_score,"ga":r.home_score,"opp":elo[h]})
    return elo,hist,np.array(X),np.array(y),pd.Series(dates),np.array(XP),np.array(yP)

FEATS=["elo_diff","perf_vs_exp_diff","perf_vs_TOP_diff","gf_diff","ga_diff",
       "opp_strength_diff","rest_diff","xG90_diff","xGA90_diff"]

def rolling(t,hist,today):
    hh=hist.get(t,[])[-10:]
    if len(hh)<5: return 0,0,1.3,1.3,1500,7
    perf=np.mean([x["pts"]-x["exp"] for x in hh])
    big=[x["pts"]-x["exp"] for x in hh if x["opp"]>=TOP_ELO]
    return (perf,(np.mean(big) if big else 0.0),np.mean([x["gf"] for x in hh]),
            np.mean([x["ga"] for x in hh]),np.mean([x["opp"] for x in hh]),
            min((today-hh[-1]["date"]).days,30))

def player_data():
    if not(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")): return {},{},"skipped — no Kaggle creds"
    try:
        subprocess.run(["kaggle","datasets","download","swaptr/fifa-wc-2026-players",
                        "-p","data/players","--unzip","-q"],check=True,timeout=300)
        pen,scorers={},{}
        for f in os.listdir("data/players"):
            if not f.endswith(".csv"): continue
            d=pd.read_csv(f"data/players/{f}")
            cols={c.lower():c for c in d.columns}
            tc=next((cols[k] for k in("team","country","nationality","nation") if k in cols),None)
            nc=next((cols[k] for k in("player","name","player_name","full_name") if k in cols),None)
            gc=next((cols[k] for k in cols if "goal" in k and "against" not in k),None)
            ic=next((cols[k] for k in cols if "injur" in k or "status" in k),None)
            rc=next((cols[k] for k in cols if "red" in k),None)
            if not tc: continue
            for _,row in d.iterrows():
                t=NAME_FIX.get(str(row[tc]),str(row[tc]))
                if t not in ALL_TEAMS: continue
                p=0
                if ic and isinstance(row.get(ic),str) and any(w in row[ic].lower() for w in("injur","out","doubt")): p+=15
                if rc and pd.notna(row.get(rc)) and float(row.get(rc) or 0)>=1: p+=10
                if p: pen[t]=min(pen.get(t,0)+p,60)
                if nc and gc and pd.notna(row.get(gc)) and float(row.get(gc) or 0)>0:
                    scorers.setdefault(t,[]).append((str(row[nc]),float(row[gc])))
        for t in scorers: scorers[t]=sorted(scorers[t],key=lambda x:-x[1])[:6]
        return pen,scorers,f"applied to {len(pen)} teams; scorer data for {len(scorers)} teams"
    except Exception as e: return {},{},f"failed ({type(e).__name__})"

def main():
    df,fd_note=load(); xf=load_xg()
    elo,hist,X,y,dts,XP,yP=build(df,xf)
    tr=(dts<TRAIN_UNTIL).values; te=((dts>=WC_GROUP_START)&(dts<=WC_GROUP_END)).values
    def brier(p,yt):
        oh=np.zeros_like(p); oh[np.arange(len(yt)),yt]=1; return float(np.mean(np.sum((p-oh)**2,1)))

    # ── MODEL ZOO (v4: + ExtraTrees, BaggedTrees, HistGB, XGBoost, soft-voting Ensemble) ──
    cands={
      "LogisticRegression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=0.5)),
      "LDA":make_pipeline(StandardScaler(),LinearDiscriminantAnalysis()),
      "RandomForest":RandomForestClassifier(n_estimators=600,min_samples_leaf=20,
          max_features="sqrt",random_state=0,n_jobs=-1),
      "ExtraTrees":ExtraTreesClassifier(n_estimators=600,min_samples_leaf=20,
          max_features="sqrt",random_state=0,n_jobs=-1),
      "BaggedTrees":BaggingClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=15),
          n_estimators=300,max_samples=0.8,random_state=0,n_jobs=-1),
      "GradientBoosting":GradientBoostingClassifier(random_state=0),
      "HistGradBoost":HistGradientBoostingClassifier(learning_rate=0.05,max_iter=400,
          l2_regularization=1.0,early_stopping=True,random_state=0),
      "KNN(k=75)":make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=75,weights="distance")),
      "SVM-RBF":make_pipeline(StandardScaler(),SVC(probability=True,C=1.0,random_state=0)),
      "NeuralNet-MLP":make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,16),
          max_iter=800,early_stopping=True,random_state=0)),
    }
    if HAS_XGB:
        cands["XGBoost"]=XGBClassifier(n_estimators=500,learning_rate=0.05,max_depth=4,
            subsample=0.8,colsample_bytree=0.8,reg_lambda=2.0,objective="multi:softprob",
            eval_metric="mlogloss",random_state=0,n_jobs=-1)
    ens_members=[("lr",cands["LogisticRegression"]),("lda",cands["LDA"]),("hgb",cands["HistGradBoost"])]
    if HAS_XGB: ens_members.append(("xgb",cands["XGBoost"]))
    cands["Ensemble(soft-vote)"]=VotingClassifier(estimators=[(n,m) for n,m in ens_members],voting="soft")

    comp,fitted=[],{}
    for name,m in cands.items():
        m.fit(X[tr],y[tr]); p=m.predict_proba(X[te])
        comp.append({"model":name,"holdout_matches":int(te.sum()),
          "accuracy":round(accuracy_score(y[te],p.argmax(1))*100,1),
          "log_loss":round(log_loss(y[te],p,labels=[0,1,2]),4),"brier":round(brier(p,y[te]),4)})
        fitted[name]=m

    # ── FIXED Poisson goals model: scaled pipeline, lighter alpha ──
    pr=make_pipeline(StandardScaler(),PoissonRegressor(alpha=0.05,max_iter=3000))
    trP=np.repeat(tr,2)
    pr.fit(XP[trP],yP[trP])
    idx=np.where(te)[0]; probsP=[]
    for i in idx:
        lh=max(.2,pr.predict(XP[2*i:2*i+1])[0]); la=max(.2,pr.predict(XP[2*i+1:2*i+2])[0])
        M=np.outer([pois.pmf(k,lh) for k in range(9)],[pois.pmf(k,la) for k in range(9)])
        probsP.append([np.triu(M,1).sum(),np.trace(M),np.tril(M,-1).sum()])
    pP=np.array(probsP)
    comp.append({"model":"PoissonGoals(ref)","holdout_matches":int(te.sum()),
      "accuracy":round(accuracy_score(y[te],pP.argmax(1))*100,1),
      "log_loss":round(log_loss(y[te],pP,labels=[0,1,2]),4),"brier":round(brier(pP,y[te]),4)})

    best=min([c for c in comp if "ref" not in c["model"]],key=lambda c:c["log_loss"])
    model=fitted[best["model"]]; model.fit(X,y)
    pr.fit(XP,yP)
    pen,scorers,pen_note=player_data()
    eloA={t:elo.get(t,1500)-pen.get(t,0) for t in ALL_TEAMS}
    today=pd.Timestamp.now()

    def P(a,b):
        fa,fb=rolling(a,hist,today),rolling(b,hist,today)
        xga_,xgaa_=xg_at(xf,a,today); xgb_,xgab_=xg_at(xf,b,today)
        row=[[eloA[a]-eloA[b],fa[0]-fb[0],fa[1]-fb[1],fa[2]-fb[2],fa[3]-fb[3],
              fa[4]-fb[4],fa[5]-fb[5],xga_-xgb_,xgaa_-xgab_]]
        pl,pd_,pw=model.predict_proba(row)[0]; tot=pw+pl
        return round((pw+pd_*pw/tot)*100,1),round((pl+pd_*pl/tot)*100,1)

    def detail(a,b):
        fa,fb=rolling(a,hist,today),rolling(b,hist,today)
        xga_,xgaa_=xg_at(xf,a,today); xgb_,xgab_=xg_at(xf,b,today)
        lh=max(.2,pr.predict([[eloA[a]-eloA[b],fa[2],fb[3],xga_,xgab_]])[0])
        la=max(.2,pr.predict([[eloA[b]-eloA[a],fb[2],fa[3],xgb_,xgaa_]])[0])
        M=np.outer([pois.pmf(k,lh) for k in range(7)],[pois.pmf(k,la) for k in range(7)])
        M/=M.sum()
        top=sorted([(h,aa,float(M[h,aa])) for h in range(7) for aa in range(7)],key=lambda x:-x[2])[:6]
        def sc(team,lam):
            out=[]
            tot=sum(x[1] for x in scorers.get(team,[])) or 1
            for name,g in scorers.get(team,[]):
                out.append({"player":name,"pScore":round((1-math.exp(-lam*g/tot))*100,1)})
            return out
        return {"xgHome":round(float(lh),2),"xgAway":round(float(la),2),
          "topScorelines":[{"h":h,"a":aa,"p":round(p*100,1)} for h,aa,p in top],
          "over25":round(float(sum(M[h,aa] for h in range(7) for aa in range(7) if h+aa>=3))*100,1),
          "btts":round(float(sum(M[h,aa] for h in range(1,7) for aa in range(1,7)))*100,1),
          "scorersHome":sc(a,lh),"scorersAway":sc(b,la),
          "eloHome":round(eloA[a]),"eloAway":round(eloA[b])}

    wc=df[(df.date>=WC_GROUP_START)&(df.date<=WC_GROUP_END)&
          (df.home_team.isin(ALL_TEAMS))&(df.away_team.isin(ALL_TEAMS))]
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

    def enrich(m):
        if m["home"] and m["away"]:
            m["detail"]=detail(m["home"],m["away"])
            w=winner_of(m)
            if not w:
                m["pHome"],m["pAway"]=P(m["home"],m["away"])
                m["predicted"]=m["home"] if m["pHome"]>=m["pAway"] else m["away"]
            else: m["winner"]=w
        return m
    pool={}
    r32=[enrich(dict(m)) for m in R32]
    for m in r32: pool[m["id"]]=m.get("winner") or m.get("predicted")
    def rnd(feeds):
        out=[]
        for mid,f1,f2,date in feeds:
            m=enrich({"id":mid,"date":date,"home":pool.get(f1),"away":pool.get(f2),"status":"SCHEDULED"})
            pool[mid]=m.get("winner") or m.get("predicted"); out.append(m)
        return out
    r16=rnd(R16_FEEDS); qf=rnd(QF_FEEDS); sf=rnd(SF_FEEDS); fin=rnd([FN_FEED])[0]

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

    json.dump({"generated_at":datetime.now(timezone.utc).isoformat(),
      "methodology":{"candidates_compared":comp,"selected_model":best["model"],
        "selection_rule":"lowest log-loss on 2026 WC group-stage holdout",
        "features":FEATS,"train_window":f"{TRAIN_FROM} → {TRAIN_UNTIL}",
        "player_adjustment":pen_note,"football_data_org":fd_note,
        "statsbomb_xg":"loaded" if xf is not None else "not present"},
      "standings":standings,"bracket":{"r32":r32,"r16":r16,"qf":qf,"sf":sf,"final":fin},
      "champion_probs":champ_probs},open(OUT,"w"),indent=1)
    print("OK | best:",best["model"],best["log_loss"])

if __name__=="__main__": main()

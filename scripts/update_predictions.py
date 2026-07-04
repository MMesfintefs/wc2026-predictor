"""WC2026 Predictor v3 — model zoo + match detail + football-data.org autofill."""
import json, math, os, subprocess
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from scipy.stats import poisson as pois

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

# ── football-data.org autofill (optional, FOOTBALL_DATA_TOKEN secret) ──
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
    if not(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")): return

#!/usr/bin/env python3
"""
GAM3ARCH v3.2-dev — Agent-Based Simulation Framework
Supports presets: paper / realistic
"""
import os, json, time, argparse
import numpy as np, pandas as pd
from dataclasses import dataclass, asdict
from scipy.stats import sem

STATE_NAMES = ["Forge", "Nexus", "Back", "Horizon"]
STATE_IDX = {n:i for i,n in enumerate(STATE_NAMES)}
P_BASE = np.array([[0.55,0.15,0.20,0.10],
                   [0.20,0.50,0.20,0.10],
                   [0.25,0.25,0.40,0.10],
                   [0.30,0.20,0.10,0.40]])

def F_mult(fat,F50,p): return 1.0/(1.0+(fat/F50)**p)
def M_mult(mot,k_m,M_max): return 1.0+k_m*(mot/M_max)
def H_mult(hor,k_h): return 1.0+k_h*hor
def normalize_rows(mat): 
    s=mat.sum(1,keepdims=True); s[s==0]=1; return mat/s
def sample_states(probs):
    c=np.cumsum(probs,axis=1); r=np.random.rand(probs.shape[0],1)
    return np.argmax(c>=r,axis=1)

@dataclass
class Config:
    N:int; T:int; runs:int; seed:int
    R_max:float; s_n:float; k_m:float; k_h:float; M_max:float
    F50:float; p:float; alpha:float; beta:float; gamma:float; delta:float
    F_burn:float; burn_window:int; out_dir:str

def load_preset(name:str)->Config:
    presets={
        "paper": dict(N=500,T=500,runs=20,seed=42,R_max=1.0,s_n=5.0,k_m=1.2,k_h=1.0,M_max=1.0,
                      F50=50.0,p=2.0,alpha=0.8,beta=0.5,gamma=0.4,delta=0.6,F_burn=0.8,burn_window=50,out_dir="results"),
        "realistic": dict(N=1000,T=1000,runs=5,seed=42,R_max=1.0,s_n=0.05,k_m=1.2,k_h=1.0,M_max=1.0,
                      F50=50.0,p=2.0,alpha=0.03,beta=0.5,gamma=0.4,delta=0.6,F_burn=0.75,burn_window=10,out_dir="results")
    }
    return Config(**presets[name])

class GAM3ARCHSim:
    def __init__(self,cfg:Config): self.cfg=cfg
    def _init(self,rng):
        N=self.cfg.N
        return dict(state=rng.choice(4,size=N,p=[0.5,0.2,0.2,0.1]),
                    Fat=rng.random(N)*0.2,Mot=rng.random(N)*0.8,
                    Hor=np.zeros(N),S=rng.random(N)*0.3,
                    burnout=np.zeros(N,bool))
    def _compute(self,B): return normalize_rows(P_BASE*B)
    def run_single(self,B,rng):
        cfg,N,T=self.cfg,self.cfg.N,self.cfg.T
        P=self._compute(B); pop=self._init(rng); window=np.zeros((N,cfg.burn_window)); wi=0
        for t in range(T):
            rec=(pop["state"]==STATE_IDX["Back"]).astype(float)
            pop["Fat"]=np.clip(pop["Fat"]+cfg.alpha-cfg.beta*rec+rng.normal(0,0.02,N),0,None)
            pop["Mot"]=np.clip(pop["Mot"]+cfg.gamma*rng.random(N)-cfg.delta*pop["Fat"]+rng.normal(0,0.02,N),0,None)
            pop["Hor"]=(pop["state"]==STATE_IDX["Horizon"])*0.7
            Res=cfg.R_max*(1+cfg.s_n*pop["S"])*F_mult(pop["Fat"],cfg.F50,cfg.p)*M_mult(pop["Mot"],cfg.k_m,cfg.M_max)*H_mult(pop["Hor"],cfg.k_h)
            pop["state"]=sample_states(P[pop["state"]])
            window[:,wi%cfg.burn_window]=pop["Fat"]; wi+=1
            if t>=cfg.burn_window: pop["burnout"]|=(window.mean(1)>cfg.F_burn)
        return dict(burnout=float(pop["burnout"].mean()),resonance=float(Res.mean()))
    def run(self,scenarios):
        os.makedirs(self.cfg.out_dir,exist_ok=True)
        rng=np.random.default_rng(self.cfg.seed); results=[]
        for name,B in scenarios.items():
            vals=[self.run_single(B,np.random.default_rng(rng.integers(0,2**32))) for _ in range(self.cfg.runs)]
            burnout=[v["burnout"] for v in vals]; res=[v["resonance"] for v in vals]
            results.append(dict(scenario=name,burnout_mean=np.mean(burnout),burnout_sem=sem(burnout),
                                resonance_mean=np.mean(res),resonance_sem=sem(res)))
        df=pd.DataFrame(results); df.to_csv(f"{self.cfg.out_dir}/summary.csv",index=False)
        print(df.round(4)); return df

def main():
    p=argparse.ArgumentParser(); p.add_argument("--preset",choices=["paper","realistic"],default="paper")
    a=p.parse_args(); cfg=load_preset(a.preset); sim=GAM3ARCHSim(cfg)
    scenarios={"Baseline":np.ones((4,4)),"StrongBridges":np.ones((4,4))*0.98}
    B=np.ones((4,4))*0.95; B[0,2]=0.1; B[3,0]=0.2; scenarios["WeakBridges"]=B
    sim.run(scenarios)

if __name__=="__main__": main()

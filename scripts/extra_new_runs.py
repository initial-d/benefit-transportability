import json, math, sys
from pathlib import Path
import numpy as np
src=(Path(__file__).resolve().parent/'analyze_new_runs.py').read_text().split('R = {}')[0]
_args=sys.argv[:]
sys.argv=['x',_args[1] if len(_args)>1 else '.', _args[1] if len(_args)>1 else '.','/dev/null']
exec(src)
exp=Path(sys.argv[2]) if len(sys.argv)>2 and sys.argv[2]!='/dev/null' else Path('.'); X={}
audit_items=json.loads((exp/'data/forge_f/f_audit_items.json').read_text()); test_items=json.loads((exp/'data/forge_f/f_test_items.json').read_text())
Fa_all=load_new(exp/'results/forge_f/f_query_metrics.jsonl','fast2','slow',[x['id'] for x in audit_items]); Ft=load_new(exp/'results/forge_f/f_query_metrics.jsonl','fast2','slow',[x['id'] for x in test_items])
m=np.zeros(len(Fa_all.fam),bool)
for f in FAM: m[np.where(Fa_all.fam==f)[0][:20]]=True
Fa=Fa_all.sub(m)
for g in [0.05,0.10,0.15]:
    fm=threshold_fast(fam_tau(Fa),g); e=evaluate_family(Ft,fm)
    X[f'F_unconstrained_g{g}']={'fast':[f for f,mm in zip(FAM,fm) if mm],**e,'random_same_rate':random_matched(Ft,e['slow_rate'])['acc']}
for a in [0.8,0.6,0.4]:
    k={0.8:1,0.6:2,0.4:3}[a]
    fm=one_fast(fam_tau(Fa),np.maximum(fam_dk(Fa),0),GAMMA,k); e=evaluate_family(Ft,fm)
    X[f'F_alpha{a}']={'fast':[f for f,mm in zip(FAM,fm) if mm],**e,'random_same_rate':random_matched(Ft,e['slow_rate'])['acc']}
# historical rule (A-E) unconstrained
main_pair=load_main(Path(sys.argv[1])); hist=main_pair.envs('A','B','C','D','E_audit','E_test')
fm=threshold_fast(fam_tau(hist)); e=evaluate_family(Ft,fm); X['F_hist_unconstrained']={'fast':[f for f,mm in zip(FAM,fm) if mm],**e}
# bootstrap of unconstrained rule
rng=np.random.default_rng(3); by={f:np.where(Fa.fam==f)[0] for f in FAM}; acc=[];sav=[]
for _ in range(5000):
    idx=np.concatenate([by[f][rng.integers(0,20,20)] for f in FAM])
    a=Pair(Fa.name,Fa.env[idx],Fa.fam[idx],Fa.X[idx],Fa.yl[idx],Fa.yu[idx],Fa.kl[idx],Fa.ku[idx],Fa.ids[idx])
    e=evaluate_family(Ft,threshold_fast(fam_tau(a))); acc.append(e['acc']); sav.append(e['save_vs_slow'])
X['F_unconstrained_bootstrap']={'acc_mean':float(np.mean(acc)),'acc_p05':pct(acc,5),'save_mean':float(np.mean(sav)),'save_p05':pct(sav,5)}
# CI for unconstrained gamma .15
fm=threshold_fast(fam_tau(Fa)); isf=np.array([dict(zip(FAM,fm))[f] for f in Ft.fam]); y=np.where(isf,Ft.yl,Ft.yu); k=np.where(isf,Ft.kl,Ft.ku)
bi=rng.integers(0,len(y),(10000,len(y))); X['F_unconstrained_ci']={'acc_cp':clopper_pearson(int(y.sum()),len(y)),'save_ci':[pct(1-k[bi].sum(1)/Ft.ku[bi].sum(1),2.5),pct(1-k[bi].sum(1)/Ft.ku[bi].sum(1),97.5)]}
# drift arms: adaptive audit and unconstrained
e_items=json.loads((Path(sys.argv[1])/'data/forge/e_items.json').read_text()); split={x['id']:('audit' if '.calibration.' in x['id'] else 'test') for x in e_items}
ref=load_new(exp/'results/drift_e/E_rerun_t0.jsonl','fast2','slow'); ra=ref.sub(np.array([split[i]=='audit' for i in ref.ids]))
ref_unc=threshold_fast(fam_tau(ra))
for name,path,lo,up in [('rerun_t0','results/drift_e/E_rerun_t0.jsonl','fast2','slow'),('temperature_0.7','results/drift_e/E_rerun_t07.jsonl','fast2_t07','slow_t07'),('prompt_template','results/drift_e/E_rerun_tpl2.jsonl','fast2_tpl2','slow_tpl2')]:
    P=load_new(exp/path,lo,up); ma=np.array([split[i]=='audit' for i in P.ids]); A,T=P.sub(ma),P.sub(~ma)
    taus=fam_tau(A); dk=np.maximum(fam_dk(A),0)
    # adaptive: CI straddling -> pool unavailable beyond 20 in E audit; use 90% CI and fall back to slow for straddling families
    lohi=[(lambda b:(b.mean()-1.645*max(b.std(ddof=1),1/math.sqrt(len(b)))/math.sqrt(len(b)), b.mean()+1.645*max(b.std(ddof=1),1/math.sqrt(len(b)))/math.sqrt(len(b))))(A.b[A.fam==f]) for f in FAM]
    safe=np.array([hi<GAMMA or (lo<=GAMMA<=hi and t<=GAMMA-0.05) for (lo,hi),t in zip(lohi,taus)])
    cons=one_fast(np.where(safe,taus,np.inf),dk)
    X[f'drift_{name}']={'tau_audit':dict(zip(FAM,map(float,taus))),
        'frozen_unconstrained':{'fast':[f for f,mm in zip(FAM,ref_unc) if mm],**evaluate_family(T,ref_unc)},
        'reaudit_unconstrained':{'fast':[f for f,mm in zip(FAM,threshold_fast(taus)) if mm],**evaluate_family(T,threshold_fast(taus))},
        'reaudit_conservative':{'fast':[f for f,mm in zip(FAM,cons) if mm],**evaluate_family(T,cons)},
        'uniform_slow':float(T.yu.mean())}
Path(_args[2] if len(_args)>2 else 'out/new_runs_extra.json').write_text(json.dumps(X,indent=1,default=float))
for k,v in X.items():
    if isinstance(v,dict) and 'acc' in v: print(k,v.get('fast'),round(v['acc'],3),v.get('ff'),round(v.get('save_vs_slow',0),3),round(v.get('random_same_rate',0),3))
    else: print(k,json.dumps(v,default=float)[:600])

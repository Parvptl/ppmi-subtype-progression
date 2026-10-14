import os, sys
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
import numpy as np, importlib
from scipy.stats import skew, kurtosis
m=importlib.import_module("26_clusterability_extended")
from ppmi_subtype.config import load_config
from ppmi_subtype import timevarying as TV
cfg=load_config(); data=m.load_data(cfg)
fit=TV.fit(data,knots=(),random_slopes=True)
def summ(d,f=None):
    f=f or TV.fit(d,knots=(),random_slopes=True)
    S=m.blup_slopes(d,f); I=np.array([TV.blup(f,d.times[p],d.responses[p])[0][:2] for p in d.patients])
    Y=np.concatenate([d.responses[p] for p in d.patients])
    allz=np.mean([np.all(d.responses[p][:,1]==0) for p in d.patients])
    return dict(sk_sT=skew(S[:,0]),sk_sP=skew(S[:,1]),ku_sT=kurtosis(S[:,0]),ku_sP=kurtosis(S[:,1]),
                sk_iT=skew(I[:,0]),sk_iP=skew(I[:,1]),corr_s=np.corrcoef(S.T)[0,1],
                pigd0=np.mean(Y[:,1]==0),trem0=np.mean(Y[:,0]==0),allzeroP=allz)
r=summ(data,fit); print("real  ",{k:round(v,3) for k,v in r.items()})
print("Sigma_b slope sd", np.sqrt(np.diag(fit.sigma_b))[2:], "corr int/slope", np.round(fit.sigma_b/np.sqrt(np.outer(np.diag(fit.sigma_b),np.diag(fit.sigma_b))),2).tolist())
for rep in range(4):
    rng=np.random.default_rng([42,2,rep,0])
    print("meas%d "%rep,{k:round(v,3) for k,v in summ(m.simulate(data,fit,rng,measured=True)).items()})

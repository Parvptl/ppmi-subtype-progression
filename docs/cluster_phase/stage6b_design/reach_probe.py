import os, sys, importlib
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
m=importlib.import_module("29_clusterability_skewed_null")
from ppmi_subtype.config import load_config
from ppmi_subtype import timevarying as TV
cfg=load_config(); data=m.S26.load_data(cfg); fit=TV.fit(data,knots=(),random_slopes=True)
real=m.shape(data,fit)
print("real", {k:round(real[k],2) for k in ["skew_slope_pigd","kurt_slope_pigd","skew_slope_tremor","kurt_slope_tremor","skew_int_pigd","share_pigd_zero_visits"]})
grid=[(d,nu) for d in (0.6,0.9,0.99) for nu in (3.0,4.0,6.0)]
tasks=[(r,np.full(4,d),nu) for d,nu in grid for r in range(6)]
with ProcessPoolExecutor(54, initializer=m._S.update, initargs=({"data":data,"fit":fit},)) as pool:
    out=list(pool.map(m._cal_task,tasks))
df=pd.DataFrame(out); df["delta"]=[t[1][0] for t in tasks]; df["nu"]=[t[2] for t in tasks]
print(df.groupby(["delta","nu"])[["skew_slope_pigd","kurt_slope_pigd","skew_slope_tremor","kurt_slope_tremor","skew_int_pigd","share_pigd_zero_visits"]].mean().round(2).to_string())

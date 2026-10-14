import os, sys, importlib, socket, json
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
import pandas as pd
m=importlib.import_module("26_clusterability_extended")
from ppmi_subtype.config import load_config
row=m.evaluate(m.load_data(load_config()), m.SEED)
old=pd.read_csv("experiments/20261014T100107Z_s6_clusterability/real.csv").iloc[0]
keys=["dBIC","best_K","drop1_ARI_K2","drop1_ARI_K3","drop2_ARI_K2","trunc3_ARI_K2","silhouette_K2","BIC_K1","BIC_K3"]
out={"host":socket.gethostname(),**{k:{"this_node":float(row[k]),"node004_saved":float(old[k]),"abs_diff":abs(float(row[k])-float(old[k]))} for k in keys}}
print(json.dumps(out,indent=1))

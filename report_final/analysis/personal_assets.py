#!/usr/bin/env python3
"""Build personal-Vanda tables/figures for the combined final report.
Only results/vanda_harry CSVs are read; the GitHub baseline report assets are
copied separately and are not modified.
"""
from pathlib import Path
import csv, statistics
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/vanda_harry'; T=ROOT/'report_final/tables'; F=ROOT/'report_final/figures'
T.mkdir(parents=True,exist_ok=True); F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':0.3,'axes.spines.top':False,'axes.spines.right':False,'figure.autolayout':True,'savefig.dpi':200})

def rows(p): return list(csv.DictReader(open(p)))
def med(rs,k): return statistics.median(float(r[k]) for r in rs)
def save(fig,name):
    fig.savefig(F/name); fig.savefig(F/(Path(name).stem+'.png'),dpi=130); plt.close(fig)

# E1 large
e1=rows(R/'1552578.stdct-mgmt-02_e1_serial.csv')
g=defaultdict(list)
for r in e1:
    if r.get('experiment')=='E1-E1C': g[float(r['rows'])].append(r)
with (T/'e1_large.tex').open('w') as f:
    f.write('\\begin{tabular}{rrrr}\n\\toprule\n$N$ & fit (s) & total (s) & generation (s)\\\\\n\\midrule\n')
    for n in sorted(g):
        f.write(f'{int(n):,} & {med(g[n],"fit_s"):.3f} & {med(g[n],"total_s"):.3f} & {med(g[n],"distribution_s"):.3f}\\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n')

# E3 large
e3=rows(R/'1552703.stdct-mgmt-02_e2_large.csv')
g=defaultdict(list)
for r in e3: g[int(r['ranks'])].append(r)
base=med(g[1],'fit_s')
e3data=[]
for rank in sorted(g):
    fit=med(g[rank],'fit_s'); e3data.append((rank,fit,base/fit,base/fit/rank,med(g[rank],'mpi_reduction_s')))
with (T/'e3_large.tex').open('w') as f:
    f.write('\\begin{tabular}{rrrrr}\n\\toprule\nranks & fit (s) & speedup & efficiency & collective (s)\\\\\n\\midrule\n')
    for rank,fit,sp,eff,coll in e3data:
        f.write(f'{rank} & {fit:.4f} & {sp:.2f} & {100*eff:.1f}\\% & {coll:.4f}\\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n')
fig,ax=plt.subplots(figsize=(5.6,3.6))
ranks=np.array([x[0] for x in e3data]); sp=np.array([x[2] for x in e3data]); eff=np.array([100*x[3] for x in e3data])
ax.plot(ranks,sp,'o-',color='C0',label='speedup'); ax.plot(ranks,ranks,'k--',lw=1,label='ideal')
ax.set_xscale('log',base=2); ax.set_yscale('log',base=2); ax.set_xlabel('MPI ranks'); ax.set_ylabel('speedup')
ax.set_title('10M fixed-total MPI strong scaling'); ax.legend(fontsize=7)
save(fig,'e3_large.pdf')

# E5 OpenMP
e5o=rows(R/'1557219.stdct-mgmt-02_e5_openmp_original.csv'); e5m=rows(R/'1557219.stdct-mgmt-02_e5_openmp_10m.csv')
go=defaultdict(list); gm=defaultdict(list)
for r in e5o: go[int(r['threads'])].append(r)
for r in e5m: gm[int(r['threads'])].append(r)
bo=med(go[1],'fit_s'); bm=med(gm[1],'fit_s')
e5=[]
for t in sorted(go):
    fo=med(go[t],'fit_s'); fm=med(gm[t],'fit_s'); e5.append((t,fo,bo/fo,fm,bm/fm))
with (T/'e5_openmp.tex').open('w') as f:
    f.write('\\begin{tabular}{rrrrr}\n\\toprule\nthreads & original fit (s) & speedup & 10M fit (s) & speedup\\\\\n\\midrule\n')
    for t,fo,so,fm,sm in e5:
        f.write(f'{t} & {fo:.4f} & {so:.2f} & {fm:.4f} & {sm:.2f}\\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n')
fig,ax=plt.subplots(1,2,figsize=(9.2,3.6))
ax[0].plot([x[0] for x in e5],[x[1] for x in e5],'o-',label='581,012'); ax[0].plot([x[0] for x in e5],[x[3] for x in e5],'s-',label='10,000,000')
ax[0].set_xscale('log',base=2); ax[0].set_yscale('log'); ax[0].set_xlabel('OpenMP threads'); ax[0].set_ylabel('fit (s)'); ax[0].set_title('Vanda OpenMP scaling'); ax[0].legend(fontsize=7)
ax[1].plot([x[0] for x in e5],[x[2] for x in e5],'o-',label='581,012'); ax[1].plot([x[0] for x in e5],[x[4] for x in e5],'s-',label='10,000,000'); ax[1].plot([x[0] for x in e5],[x[0] for x in e5],'k--',lw=1,label='ideal')
ax[1].set_xscale('log',base=2); ax[1].set_yscale('log',base=2); ax[1].set_xlabel('OpenMP threads'); ax[1].set_ylabel('speedup'); ax[1].set_title('OpenMP speedup'); ax[1].legend(fontsize=7)
save(fig,'e5_openmp.pdf')

# E6-64 and partial E6-256
def hyb(path):
    d=defaultdict(list)
    for r in rows(path): d[(int(r['ranks']),int(r['threads']))].append(r)
    out=[]
    for k in sorted(d,key=lambda x:(-x[0],x[1])):
        v=d[k]; out.append((k[0],k[1],med(v,'fit_s'),med(v,'local_compute_s'),med(v,'mpi_reduction_s'),med(v,'thread_reduction_s')))
    return out
h64=hyb(R/'1557220.stdct-mgmt-02_e6_hybrid_64.csv')
h256=hyb(R/'1557222.stdct-mgmt-02_e6_hybrid_256.csv')
with (T/'e6_hybrid_10m.tex').open('w') as f:
    f.write('\\begin{tabular}{rrrrrr}\n\\toprule\nbudget & ranks & threads & fit (s) & local (s) & collective (s)\\\\\n\\midrule\n')
    for r,t,fit,loc,coll,tr in h64:
        f.write(f'64 & {r} & {t} & {fit:.4f} & {loc:.4f} & {coll:.4f}\\\\\n')
    for r,t,fit,loc,coll,tr in h256:
        f.write(f'256 & {r} & {t} & {fit:.4f} & {loc:.4f} & {coll:.4f}\\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n')
fig,ax=plt.subplots(1,2,figsize=(9.2,3.6))
ax[0].bar([f'{r}x{t}' for r,t,*_ in h64],[x[2] for x in h64],color='C0'); ax[0].set_title('10M Hybrid, 64 cores'); ax[0].tick_params(axis='x',rotation=45); ax[0].set_ylabel('fit (s)')
ax[1].bar([f'{r}x{t}' for r,t,*_ in h256],[x[2] for x in h256],color='C3'); ax[1].set_title('10M Hybrid, 256-core partial (no 256x1)'); ax[1].tick_params(axis='x',rotation=45); ax[1].set_ylabel('fit (s)')
save(fig,'e6_hybrid_10m.pdf')

# E7
e7=[]
for label in ('core','ppr32','ppr16'):
    d=defaultdict(list)
    for r in rows(R/f'1557223.stdct-mgmt-02_e7_{label}.csv'): d[int(r['ranks'])].append(r)
    for rank in sorted(d): e7.append((label,rank,med(d[rank],'fit_s'),med(d[rank],'mpi_reduction_s'),med(d[rank],'local_compute_s')))
with (T/'e7_layout.tex').open('w') as f:
    f.write('\\begin{tabular}{llrrr}\n\\toprule\nmapping & ranks & fit (s) & collective (s) & local (s)\\\\\n\\midrule\n')
    for label,rank,fit,coll,loc in e7:
        f.write(f'{label} & {rank} & {fit:.4f} & {coll:.4f} & {loc:.4f}\\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n')

# recovery status (static provenance, no result rows)
with (T/'recovery_status.tex').open('w') as f:
    f.write('''\\begin{tabular}{lllll}\n\\toprule
Experiment & Job & Resource & State & Dependency\\\\
\\midrule
E2 1-rank recovery & 1563234 & 1 core & queued & none\\\\
E6-256 256$\\times$1 recovery & 1563235 & 4$\\times$64 & queued & none\\\\
E6-128 backup smoke & 1563236 & 2$\\times$64 & queued & none\\\\
E6-128 backup production & 1563237 & 2$\\times$64 & held & afterok:1563236\\\\
\\bottomrule\n\\end{tabular}\n''')
print('personal assets built')

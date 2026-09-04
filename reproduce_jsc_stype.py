from core import *

def main(out='output'):
    root=Path(out); d=root/'data'; f=root/'figures'; d.mkdir(parents=True,exist_ok=True); f.mkdir(parents=True,exist_ok=True)

    # A. Exact single-layer asymptotics check.
    rows=[]
    for z in [0.5,0.25,0.125,0.0625,0.03125,0.015625]:
        I=frozen_I(z); K=frozen_K(z)
        rows.append({'z':z,'I(z)':I,'I/(z^2/24)':I/(z*z/24),'K(z)':K,'K/(73z^4/11520)':K/(73*z**4/11520)})
    pd.DataFrame(rows).to_csv(d/'balanced_single_layer_asymptotics.csv',index=False)

    # B. Full BVP norm crossover, same total N.
    cross=[]
    for ratio in [1,2,4,10,100]:
        for eps in [1e-3,1e-5,1e-7]:
            for N in [128,256,512,1024]:
                for rule in ['sym','max','bal']:
                    x,U,split=solve_manufactured(N,eps,float(ratio),1.0,rule)
                    enodal,erecon,ebal,eL2,eD=error_metrics(x,U,eps,float(ratio),1.0)
                    cross.append({'ratio':ratio,'epsilon':eps,'N':N,'rule':rule,'NL':split[0],'NR':split[2],
                                  'nodal_max_error':enodal,'reconstructed_Linf_error':erecon,
                                  'balanced_error':ebal,'L2_error':eL2,'balanced_derivative':eD})
    cross=pd.DataFrame(cross); cross.to_csv(d/'full_bvp_norm_crossover.csv',index=False)

    # Winner rates between sqrt and 2/3 rules (exclude symmetric ratio=1 ties).
    wins=[]
    for (ratio,eps,N),g in cross[cross.ratio>1].groupby(['ratio','epsilon','N']):
        gm=g.set_index('rule')
        em=gm.loc['max','nodal_max_error']; eb=gm.loc['bal','nodal_max_error']
        bm=gm.loc['max','balanced_error']; bb=gm.loc['bal','balanced_error']
        mtol=1e-13*max(1.0,abs(em),abs(eb)); btol=1e-13*max(1.0,abs(bm),abs(bb))
        max_winner='tie' if abs(em-eb)<=mtol else ('max' if em<eb else 'bal')
        bal_winner='tie' if abs(bb-bm)<=btol else ('bal' if bb<bm else 'max')
        wins.append({'ratio':ratio,'epsilon':eps,'N':N,
                     'max_winner':max_winner,'bal_winner':bal_winner,
                     'max_gain_sqrt_vs_bal_pct':100*(1-em/eb),
                     'bal_gain_23_vs_sqrt_pct':100*(1-bb/bm)})
    wins=pd.DataFrame(wins); wins.to_csv(d/'norm_crossover_winners.csv',index=False)

    # C. Exhaustive full-BVP optimum for selected cases.
    exh=[]
    for ratio in [2,4,10,100]:
        eps=1e-5
        for N in [64,128,256]:
            B=N//2; mincount=resolving_floor(N,zeta=0.7)
            records=[]
            for NL in range(mincount,B-mincount+1):
                x,_,_,_,_,_=make_mesh_given_split(N,eps,NL)
                U=solve_fd(x,eps,lambda y:1+y,1+ratio,3.0)
                enodal,erecon,eb,*_=error_metrics(x,U,eps,float(ratio),1.0)
                records.append((NL,enodal,eb))
            arr=np.array(records,float)
            nmax=int(arr[np.argmin(arr[:,1]),0]); nbal=int(arr[np.argmin(arr[:,2]),0])
            nmaxpred=constrained_split_max(ratio,1,1,1,B,N,zeta=0.7)[0]; nbalpred=constrained_split_bal(ratio,1,1,1,B,N,zeta=0.7)[0]
            exh.append({'ratio':ratio,'N':N,'max_pred_NL':nmaxpred,'max_exhaustive_NL':nmax,
                        'max_distance':abs(nmaxpred-nmax),'bal_pred_NL':nbalpred,'bal_exhaustive_NL':nbal,
                        'bal_distance':abs(nbalpred-nbal)})
    exh=pd.DataFrame(exh); exh.to_csv(d/'full_bvp_exhaustive_optima.csv',index=False)

    # D. Random frozen balanced optimum with both amplitude and decay-rate asymmetry.
    rng=np.random.default_rng(20260902); fr=[]
    for cid in range(800):
        muL=10**rng.uniform(-0.7,0.7); muR=1.0
        kL=10**rng.uniform(-0.5,0.5); kR=1.0
        eps=10**rng.uniform(-7,-3)
        N=int(rng.choice([128,256,512,1024,2048,4096])); B=N//2; c=2*math.log(N)
        mincount=resolving_floor(N,zeta=0.7)
        if 2*mincount>=B: continue
        pred=constrained_split_bal(muL,muR,kL,kR,B,N,zeta=0.7)[0]
        if not(mincount<=pred<=B-mincount): continue
        vals=[]
        for n in range(mincount,B-mincount+1):
            zL=c/n; zR=c/(B-n)
            val=frozen_bal_sq(muL,kL,eps,zL)+frozen_bal_sq(muR,kR,eps,zR)
            vals.append((n,val))
        nopt,vopt=min(vals,key=lambda t:t[1]); vpred=dict(vals)[pred]
        fr.append({'case':cid,'mu_ratio':muL/muR,'kappa_ratio':kL/kR,'epsilon':eps,'N':N,
                   'pred_NL':pred,'opt_NL':nopt,'distance':abs(pred-nopt),'relative_regret_pct':100*(math.sqrt(vpred/vopt)-1)})
    fr=pd.DataFrame(fr); fr.to_csv(d/'frozen_balanced_random_validation.csv',index=False)

    # E. Leading Pareto trade-off example ratio=10, kappa equal, B=128.
    ratio=10.; B=128
    th=np.linspace(0.08,0.92,500)
    maxobj=np.maximum(ratio/th**2,1/(1-th)**2)
    bal2=ratio**2/th**2+1/(1-th)**2
    Rm=maxobj/maxobj.min(); Rb=np.sqrt(bal2/bal2.min()); Rrob=np.maximum(Rm,Rb)
    i=np.argmin(Rrob)
    pareto=pd.DataFrame({'theta':th,'max_norm_regret':Rm,'balanced_norm_regret':Rb,'worst_regret':Rrob})
    pareto.to_csv(d/'pareto_ratio10.csv',index=False)
    summary={'pareto_theta_ratio10':float(th[i]),'pareto_worst_regret_ratio10':float(Rrob[i]),
             'sqrt_theta_ratio10':math.sqrt(ratio)/(math.sqrt(ratio)+1),
             'two_thirds_theta_ratio10':ratio**(2/3)/(ratio**(2/3)+1),
             'full_bvp_max_strict_wins':int((wins.max_winner=='max').sum()),
             'full_bvp_max_ties':int((wins.max_winner=='tie').sum()),
             'full_bvp_max_reversals':int((wins.max_winner=='bal').sum()),
             'full_bvp_bal_strict_wins':int((wins.bal_winner=='bal').sum()),
             'full_bvp_bal_ties':int((wins.bal_winner=='tie').sum()),
             'full_bvp_bal_reversals':int((wins.bal_winner=='max').sum()),
             'frozen_bal_exact_match_pct':100*float((fr.distance==0).mean()),
             'frozen_bal_within_one_pct':100*float((fr.distance<=1).mean()),
             'frozen_bal_mean_regret_pct':float(fr.relative_regret_pct.mean()),
             'frozen_bal_max_regret_pct':float(fr.relative_regret_pct.max()),
             'exhaustive_max_within1_pct':100*float((exh.max_distance<=1).mean()),
             'exhaustive_bal_within1_pct':100*float((exh.bal_distance<=1).mean())}
    (root/'summary.json').write_text(json.dumps(summary,indent=2))

    # F. Parameter-uniform pilot table.
    epsset=[10.0**(-k) for k in range(9)]
    prow=[]
    prev_eq=prev_sq=None
    for N in [32,64,128,256,512,1024,2048,4096]:
        eeq=max(pilot_error(N,e,'sym')[0] for e in epsset)
        esq=max(pilot_error(N,e,'max')[0] for e in epsset)
        req=(math.log(prev_eq/eeq,2) if prev_eq is not None else float('nan'))
        rsq=(math.log(prev_sq/esq,2) if prev_sq is not None else float('nan'))
        prow.append({'N':N,'equal_error':eeq,'equal_rate':req,'sqrt_error':esq,'sqrt_rate':rsq})
        prev_eq,prev_sq=eeq,esq
    pilot=pd.DataFrame(prow); pilot.to_csv(d/'parameter_uniform_pilot.csv',index=False)

    # G. Resolution-threshold sensitivity of the qualitative crossover.
    sens=[]
    for zeta in [0.5,0.7,0.9,1.0]:
        counts={'sqrt_strict':0,'bal_strict':0,'max_ties':0,'bal_ties':0,'max_reversals':0,'bal_reversals':0}
        for ratio in [2,4,10,100]:
            for eps in [1e-3,1e-5,1e-7]:
                for N in [128,256,512,1024]:
                    B=N//2
                    def split_z(rule):
                        m=int(math.ceil(2*math.log(N)/zeta))
                        if 2*m>=B: m=max(1,B//4)
                        if rule=='max':
                            x=B*math.sqrt(ratio)/(math.sqrt(ratio)+1); obj=lambda n:max(ratio/n**2,1/(B-n)**2)
                        else:
                            x=B*(ratio**2)**(1/3)/((ratio**2)**(1/3)+1); obj=lambda n:ratio**2/n**2+1/(B-n)**2
                        x=min(max(x,m),B-m)
                        cands=sorted(set([max(m,min(B-m,int(math.floor(x)))),max(m,min(B-m,int(math.ceil(x))))]))
                        n=min(cands,key=lambda q:(obj(q),abs(q-x))); return n
                    vals={}
                    for rule in ['max','bal']:
                        NL=split_z(rule); x,*_=make_mesh_given_split(N,eps,NL)
                        U=solve_fd(x,eps,lambda y:1+y,1+ratio,3.0)
                        nod,rec,bal,*_=error_metrics(x,U,eps,float(ratio),1.0)
                        vals[rule]=(nod,bal,NL)
                    mtol=1e-13*max(1.0,vals['max'][0],vals['bal'][0]); btol=1e-13*max(1.0,vals['max'][1],vals['bal'][1])
                    if abs(vals['max'][0]-vals['bal'][0])<=mtol: counts['max_ties']+=1
                    elif vals['max'][0]<vals['bal'][0]: counts['sqrt_strict']+=1
                    else: counts['max_reversals']+=1
                    if abs(vals['bal'][1]-vals['max'][1])<=btol: counts['bal_ties']+=1
                    elif vals['bal'][1]<vals['max'][1]: counts['bal_strict']+=1
                    else: counts['bal_reversals']+=1
        sens.append({'zeta':zeta,**counts})
    pd.DataFrame(sens).to_csv(d/'resolution_threshold_sensitivity.csv',index=False)

    summary.update({
        'pilot_N64_equal_error':float(pilot.loc[pilot.N==64,'equal_error'].iloc[0]),
        'pilot_N64_sqrt_error':float(pilot.loc[pilot.N==64,'sqrt_error'].iloc[0]),
        'resolution_sensitivity_no_reversal':bool((pd.DataFrame(sens)[['max_reversals','bal_reversals']].to_numpy()==0).all()),
    })
    (root/'summary.json').write_text(json.dumps(summary,indent=2))

    # H. Canonical Bakhvalov-S extension: pure-layer coefficients.
    st_rows=[]
    Nst=1024; epsst=1e-5; Bst=Nst//2
    cbal_theory=bakhvalov_balanced_interpolation_constant(Nst)
    for theta_frac in [0.30,0.40,0.50,0.60,0.70]:
        n=int(round(Bst*theta_frac))
        en,eb=pure_left_layer_metrics_stype(Nst,epsst,n,'bakhvalov_s')
        st_rows.append({'N':Nst,'budget_fraction':theta_frac,'NL':n,
                        'N_L^2_times_nodal_max':en*n*n,
                        'N_L_times_balanced_error':eb*n,
                        'BS_balanced_theory_constant':cbal_theory})
    stcoef=pd.DataFrame(st_rows)
    stcoef.to_csv(d/'bakhvalov_s_pure_layer_coefficients.csv',index=False)

    # I. Full BVP crossover on the Bakhvalov-S family.
    bscross=[]
    for ratio in [2,4,10,100]:
        for eps in [1e-3,1e-5,1e-7]:
            for N in [128,256,512,1024]:
                B=N//2
                nmax=constrained_split_max(ratio,1,1,1,B,N,zeta=0.7)[0]
                nbal=constrained_split_bal(ratio,1,1,1,B,N,zeta=0.7)[0]
                for rule,n in [('max',nmax),('bal',nbal)]:
                    nod,recon,bal,l2,der=manufactured_metrics_stype(N,eps,ratio,n,'bakhvalov_s')
                    bscross.append({'ratio':ratio,'epsilon':eps,'N':N,'rule':rule,'NL':n,'NR':B-n,
                                    'nodal_max_error':nod,'reconstructed_Linf_error':recon,
                                    'balanced_error':bal,'L2_error':l2,'balanced_derivative':der})
    bscross=pd.DataFrame(bscross); bscross.to_csv(d/'bakhvalov_s_norm_crossover.csv',index=False)
    bsw=[]
    for (ratio,eps,N),g in bscross.groupby(['ratio','epsilon','N']):
        q=g.set_index('rule')
        em=q.loc['max','nodal_max_error']; eb=q.loc['bal','nodal_max_error']
        bm=q.loc['max','balanced_error']; bb=q.loc['bal','balanced_error']
        mtol=1e-13*max(1.0,abs(em),abs(eb)); btol=1e-13*max(1.0,abs(bm),abs(bb))
        bsw.append({'ratio':ratio,'epsilon':eps,'N':N,
                    'max_winner':'tie' if abs(em-eb)<=mtol else ('max' if em<eb else 'bal'),
                    'bal_winner':'tie' if abs(bb-bm)<=btol else ('bal' if bb<bm else 'max'),
                    'max_gain_sqrt_vs_23_pct':100*(1-em/eb),
                    'bal_gain_23_vs_sqrt_pct':100*(1-bb/bm)})
    bsw=pd.DataFrame(bsw); bsw.to_csv(d/'bakhvalov_s_winners.csv',index=False)

    # J. Exhaustive Bakhvalov-S optima.
    bsex=[]
    for ratio in [2,4,10,100]:
        eps=1e-5; N=256; B=N//2; m=resolving_floor(N,zeta=0.7)
        vals=[]
        for n in range(m,B-m+1):
            nod,recon,bal,*_=manufactured_metrics_stype(N,eps,ratio,n,'bakhvalov_s')
            vals.append((n,nod,bal))
        arr=np.asarray(vals,float)
        noptmax=int(arr[np.argmin(arr[:,1]),0]); noptbal=int(arr[np.argmin(arr[:,2]),0])
        pmax=constrained_split_max(ratio,1,1,1,B,N,zeta=0.7)[0]
        pbal=constrained_split_bal(ratio,1,1,1,B,N,zeta=0.7)[0]
        mapv={int(r[0]):r for r in vals}
        maxreg=100*(mapv[pmax][1]/mapv[noptmax][1]-1)
        balreg=100*(mapv[pbal][2]/mapv[noptbal][2]-1)
        bsex.append({'ratio':ratio,'N':N,'sqrt_pred_NL':pmax,'max_exact_NL':noptmax,'max_distance':abs(pmax-noptmax),
                     'max_regret_pct':maxreg,'two_thirds_pred_NL':pbal,'balanced_exact_NL':noptbal,
                     'balanced_distance':abs(pbal-noptbal),'balanced_regret_pct':balreg})
    bsex=pd.DataFrame(bsex); bsex.to_csv(d/'bakhvalov_s_exhaustive_optima.csv',index=False)

    summary.update({
        'bakhvalov_balanced_theory_constant_N1024':cbal_theory,
        'bakhvalov_balanced_coeff_range_N1024':[float(stcoef['N_L_times_balanced_error'].min()),float(stcoef['N_L_times_balanced_error'].max())],
        'bakhvalov_max_coeff_range_N1024':[float(stcoef['N_L^2_times_nodal_max'].min()),float(stcoef['N_L^2_times_nodal_max'].max())],
        'bakhvalov_max_strict_wins':int((bsw.max_winner=='max').sum()),
        'bakhvalov_max_ties':int((bsw.max_winner=='tie').sum()),
        'bakhvalov_max_reversals':int((bsw.max_winner=='bal').sum()),
        'bakhvalov_bal_strict_wins':int((bsw.bal_winner=='bal').sum()),
        'bakhvalov_bal_ties':int((bsw.bal_winner=='tie').sum()),
        'bakhvalov_bal_reversals':int((bsw.bal_winner=='max').sum()),
        'bakhvalov_bal_exact_all_N256':bool((bsex.balanced_distance==0).all()),
        'bakhvalov_max_max_regret_N256_pct':float(bsex.max_regret_pct.max()),
    })
    (root/'summary.json').write_text(json.dumps(summary,indent=2))

    # Figures.
    fig,ax=plt.subplots(figsize=(5.0,3.25))
    ax.plot(stcoef.budget_fraction,stcoef['N_L^2_times_nodal_max'],marker='o',color=BLUE,label=r'$N_L^2 E_\infty^{FD}$')
    ax.plot(stcoef.budget_fraction,stcoef['N_L_times_balanced_error'],marker='s',linestyle='--',color=RED,label=r'$N_L E_{bal}^{FD}$')
    ax.axhline(cbal_theory,color=BLACK,linestyle=':',linewidth=.8,label=r'Bakhvalov-S balanced leading constant')
    ax.set_xlabel(r'Left-layer budget fraction $N_L/B$'); ax.set_ylabel('Scaled pure-layer error'); ax.legend()
    fig.tight_layout(); fig.savefig(f/'Fig6_bakhvalov_s_coefficients.pdf',bbox_inches='tight'); fig.savefig(f/'Fig6_bakhvalov_s_coefficients.png',dpi=1200,bbox_inches='tight'); plt.close(fig)

    Nmesh=256; epsmesh=1e-5; ratio_mesh=10.0
    mesh_rows=[]
    for rr,yy,cc,lab in [('sym',2.0,BLACK,'symmetric'),('max',1.0,BLUE,r'$L^\infty$ optimal'),('bal',0.0,RED,'balanced optimal')]:
        x,NL,NM,NR,tL,tR=make_mesh_custom(Nmesh,epsmesh,ratio_mesh,1.0,rule=rr)
        ax_y=np.full_like(x,yy,dtype=float); mesh_rows.append((x,ax_y,cc,lab,NL,NM,NR,tL,tR))
    fig,ax=plt.subplots(figsize=(6.2,2.1))
    for x,yy,cc,lab,NL,NM,NR,tL,tR in mesh_rows:
        ax.plot(x,yy,marker='|',linestyle='None',markersize=7,color=cc)
        ax.text(0.5,yy[0]+0.18,fr'{lab}: ${NL}|{NM}|{NR}$',ha='center',va='bottom',fontsize=7.8)
    ax.axvline(mesh_rows[0][7],color=BLACK,linestyle=':',linewidth=.7); ax.axvline(1-mesh_rows[0][8],color=BLACK,linestyle=':',linewidth=.7)
    ax.set_yticks([]); ax.set_xlabel(r'$x$'); ax.set_xlim(-.01,1.01); ax.set_ylim(-.35,2.55)
    fig.tight_layout(); fig.savefig(f/'Fig0_three_allocations_mesh.pdf',bbox_inches='tight'); fig.savefig(f/'Fig0_three_allocations_mesh.png',dpi=1200,bbox_inches='tight'); plt.close(fig)

    ratios=np.logspace(0,2,120); theta_max=np.sqrt(ratios)/(np.sqrt(ratios)+1); theta_bal=ratios**(2/3)/(ratios**(2/3)+1)
    fig,ax=plt.subplots(figsize=(4.9,3.2)); ax.semilogx(ratios,theta_max,color=BLUE,label=r'$L^\infty$: $\mu^{1/2}$ law'); ax.semilogx(ratios,theta_bal,color=RED,linestyle='--',label=r'balanced: $\mu^{2/3}$ law'); ax.set_xlabel(r'Layer-strength ratio $\mu_L/\mu_R$'); ax.set_ylabel(r'Left-layer budget fraction $N_L/B$'); ax.set_ylim(.48,.97); ax.legend(); fig.tight_layout(); fig.savefig(f/'Fig1_norm_dependent_allocations.pdf',bbox_inches='tight'); fig.savefig(f/'Fig1_norm_dependent_allocations.png',dpi=1200,bbox_inches='tight'); plt.close(fig)

    g=cross[(cross.epsilon==1e-5)&(cross.N==256)&(cross.ratio>1)].copy(); pivmax=g.pivot(index='ratio',columns='rule',values='nodal_max_error'); pivbal=g.pivot(index='ratio',columns='rule',values='balanced_error')
    fig,ax=plt.subplots(figsize=(5.0,3.25)); ax.loglog(pivmax.index,pivmax['max'],marker='o',color=BLUE,label=r'$L^\infty$, $\mu^{1/2}$ allocation'); ax.loglog(pivmax.index,pivmax['bal'],marker='s',linestyle='--',color=ORANGE,label=r'$L^\infty$, $\mu^{2/3}$ allocation'); ax.set_xlabel(r'$\mu_L/\mu_R$'); ax.set_ylabel(r'Maximum nodal error'); ax.legend(); fig.tight_layout(); fig.savefig(f/'Fig2_max_norm_crossover.pdf',bbox_inches='tight'); fig.savefig(f/'Fig2_max_norm_crossover.png',dpi=1200,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.0,3.25)); ax.loglog(pivbal.index,pivbal['max'],marker='o',color=BLUE,label=r'balanced error, $\mu^{1/2}$ allocation'); ax.loglog(pivbal.index,pivbal['bal'],marker='s',linestyle='--',color=RED,label=r'balanced error, $\mu^{2/3}$ allocation'); ax.set_xlabel(r'$\mu_L/\mu_R$'); ax.set_ylabel(r'Balanced reconstruction error'); ax.legend(); fig.tight_layout(); fig.savefig(f/'Fig3_balanced_norm_crossover.pdf',bbox_inches='tight'); fig.savefig(f/'Fig3_balanced_norm_crossover.png',dpi=1200,bbox_inches='tight'); plt.close(fig)

    fig,ax=plt.subplots(figsize=(4.2,4.0)); ax.scatter(fr.opt_NL,fr.pred_NL,s=8,facecolors='none',edgecolors=BLUE,linewidths=.6); mn=min(fr.opt_NL.min(),fr.pred_NL.min()); mx=max(fr.opt_NL.max(),fr.pred_NL.max()); ax.plot([mn,mx],[mn,mx],color=BLACK,linewidth=.8); ax.set_xlabel(r'Exact frozen balanced-optimal $N_L$'); ax.set_ylabel(r'Predicted $N_L$'); fig.tight_layout(); fig.savefig(f/'Fig4_balanced_predicted_vs_exact.pdf',bbox_inches='tight'); fig.savefig(f/'Fig4_balanced_predicted_vs_exact.png',dpi=1200,bbox_inches='tight'); plt.close(fig)

    fig,ax=plt.subplots(figsize=(5.0,3.25)); ax.plot(th,Rm,color=BLUE,label=r'$L^\infty$ regret'); ax.plot(th,Rb,color=RED,linestyle='--',label='Balanced-norm regret'); ax.plot(th,Rrob,color=BLACK,linestyle=':',label='Worst normalized regret'); ax.axvline(summary['sqrt_theta_ratio10'],color=BLUE,linestyle=':',linewidth=.8); ax.axvline(summary['two_thirds_theta_ratio10'],color=RED,linestyle=':',linewidth=.8); ax.set_xlabel(r'Allocation fraction $\theta=N_L/B$'); ax.set_ylabel('Normalized objective'); ax.set_ylim(.98,min(2.0,float(np.quantile(Rrob,.95)))); ax.legend(); fig.tight_layout(); fig.savefig(f/'Fig5_pareto_tradeoff.pdf',bbox_inches='tight'); fig.savefig(f/'Fig5_pareto_tradeoff.png',dpi=1200,bbox_inches='tight'); plt.close(fig)

    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='output', help='Output directory (default: ./output)')
    args=ap.parse_args(); main(args.out)

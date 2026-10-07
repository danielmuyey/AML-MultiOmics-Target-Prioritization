#!/usr/bin/env python3
"""
AML MULTI-OMICS TARGET PRIORITIZATION — ALL-IN-ONE ANALYSIS
===========================================================

Companion workflow for:
"A Reproducible Multi-Omics Workflow for Prioritizing Candidate
Therapeutic Targets in Acute Myeloid Leukemia"

Analyses:
  1) GSE293056 — bulk RNA-seq QC, CPM/log2(CPM+1), PCA,
     matched-day Welch tests and BH FDR.
  2) GSE300568 — CITE-seq RNA/ADT separation, per-cell QC,
     patient pseudobulk PCA, fixed-seed cell-level TruncatedSVD,
     variable RNA features and ADT summaries.
  3) GSE310778 — paired sgRosa versus sgElovl1 pseudobulk RNA/ADT
     analysis preserving mouse-level biological replication.
  4) GSE310778 — Mouse MSigDB Hallmark preranked GSEA.
  5) Supported result figures.

IMPORTANT SCIENTIFIC CONSTRAINTS
--------------------------------
- GSE293056 effect = mean log2(CPM+1) DOX - UT; it is NOT DESeq2 log2FC.
- GSE300568 TruncatedSVD is not UMAP.
- No unsupported cell-type annotation is generated.
- GSE310778 is sgElovl1 versus paired sgRosa control in an
  NRAS-mutant leukemia context; it is NOT an NRAS perturbation.
- Missing/unperformed evidence is not converted to negative evidence.
- No PPI, candidate ranking, or experimental validation is fabricated.

Example:
python AML_MultiOmics_ALL_IN_ONE_COMPLETE.py all \
  --gse293056 data/GSE293056_salmon_gene_counts.csv.gz \
  --gse300568 data/GSE300568 \
  --gse310778-prefix data/GSE310778/GSE310778 \
  --gse310778-metadata data/GSE310778/cell_metadata.csv \
  --gmt data/msigdb/mh.all.v2026.1.Mm.symbols.gmt \
  --results results --figures figures
"""

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.stats import ttest_ind, ttest_rel
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA, TruncatedSVD

SEED = 2026

def ensure_dir(path):
    p=Path(path); p.mkdir(parents=True,exist_ok=True); return p

def bh_fdr(p):
    p=np.nan_to_num(np.asarray(p,float),nan=1.0,posinf=1.0,neginf=1.0)
    return multipletests(p,method="fdr_bh")[1]

def cpm_log1p(counts):
    x=np.asarray(counts,float)
    lib=x.sum(axis=0); lib[lib==0]=1
    cpm=x/lib*1e6
    return cpm,np.log2(cpm+1)

def parse_bulk_samples(columns):
    pat=re.compile(r".*_(dox|ut)_d(\d+)_rep(\d+)$",re.I)
    rows=[]
    for s in columns:
        m=pat.fullmatch(str(s))
        if not m: raise ValueError(f"Unrecognized GSE293056 sample name: {s}")
        rows.append(dict(sample=str(s),condition=m.group(1).lower(),
                         day=int(m.group(2)),replicate=int(m.group(3))))
    return pd.DataFrame(rows)

def _pick(prefix,suffix):
    a=Path(str(prefix)+suffix+".gz")
    if a.exists(): return a
    b=Path(str(prefix)+suffix)
    if b.exists(): return b
    raise FileNotFoundError(f"Missing {prefix}{suffix}[.gz]")

def read_10x(prefix):
    prefix=Path(prefix)
    matrix=sparse.csc_matrix(mmread(str(_pick(prefix,"_matrix.mtx"))))
    features=pd.read_csv(_pick(prefix,"_features.tsv"),sep="\t",header=None)
    barcodes=pd.read_csv(_pick(prefix,"_barcodes.tsv"),sep="\t",header=None)[0].astype(str)
    if features.shape[1]>=3:
        features.columns=["feature_id","feature_name","feature_type"]+list(features.columns[3:])
    else:
        features.columns=["feature_id","feature_name"]
        features["feature_type"]="Gene Expression"
    if matrix.shape!=(len(features),len(barcodes)):
        raise ValueError("10x dimensions do not agree")
    return matrix,features,barcodes

def discover_10x(directory):
    directory=Path(directory); ans=[]
    for p in list(directory.rglob("*_matrix.mtx.gz"))+list(directory.rglob("*_matrix.mtx")):
        s=str(p)
        if s.endswith(".gz"): s=s[:-3]
        ans.append(Path(s[:-len("_matrix.mtx")]))
    return sorted(set(ans))

# ---------------------------------------------------------------------
# GSE293056
# ---------------------------------------------------------------------
def analyze_gse293056(count_file,outdir):
    out=ensure_dir(outdir)
    df=pd.read_csv(count_file)
    if not {"gene_id","gene_name"}.issubset(df.columns):
        if len(df.columns)<3: raise ValueError("GSE293056 table needs gene ID/name plus sample columns.")
        df=df.rename(columns={df.columns[0]:"gene_id",df.columns[1]:"gene_name"})
    samples=[c for c in df.columns if c not in ["gene_id","gene_name"]]
    meta=parse_bulk_samples(samples)
    x=df[samples].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(float)
    cpm,logcpm=cpm_log1p(x)

    qc=meta.copy()
    qc["library_size"]=x.sum(0)
    qc["detected_genes"]=(x>0).sum(0)
    qc["zero_fraction"]=(x==0).mean(0)
    qc.to_csv(out/"sample_qc.csv",index=False)

    keep=(cpm>1).sum(1)>=2
    model=PCA(n_components=min(5,len(samples))).fit(logcpm[keep].T)
    pcs=model.transform(logcpm[keep].T)
    scores=pd.concat([meta.reset_index(drop=True),
                      pd.DataFrame(pcs,columns=[f"PC{i+1}" for i in range(pcs.shape[1])])],axis=1)
    scores.to_csv(out/"pca_scores.csv",index=False)
    pd.DataFrame({"PC":np.arange(1,len(model.explained_variance_ratio_)+1),
                  "variance_ratio":model.explained_variance_ratio_}).to_csv(out/"pca_variance.csv",index=False)

    tests=[]; summary=[]
    days=sorted(set(meta.loc[meta.condition=="dox","day"]) &
                set(meta.loc[meta.condition=="ut","day"]))
    for day in days:
        ia=meta.index[(meta.condition=="dox")&(meta.day==day)].to_numpy()
        ib=meta.index[(meta.condition=="ut")&(meta.day==day)].to_numpy()
        tested=(cpm[:,np.r_[ia,ib]]>1).sum(1)>=2
        y=logcpm[tested]
        stat,pv=ttest_ind(y[:,ia],y[:,ib],axis=1,equal_var=False,nan_policy="omit")
        pv=np.nan_to_num(pv,nan=1.0)
        effect=y[:,ia].mean(1)-y[:,ib].mean(1)
        tab=pd.DataFrame({
            "gene_id":df.loc[tested,"gene_id"].values,
            "gene_name":df.loc[tested,"gene_name"].values,
            "day":day,
            "effect_mean_log2CPM_DOX_minus_UT":effect,
            "welch_t":stat,
            "pvalue":pv})
        tab["padj_bh"]=bh_fdr(pv)
        tab["passes_combined_threshold"]=(tab.padj_bh<0.05)&(
            tab.effect_mean_log2CPM_DOX_minus_UT.abs()>=1)
        tests.append(tab)
        summary.append({"day":day,"genes_tested":len(tab),
                        "combined_pass":int(tab.passes_combined_threshold.sum())})

    de=pd.concat(tests,ignore_index=True)
    de.to_csv(out/"matched_day_welch_tests.csv",index=False)
    pd.DataFrame(summary).to_csv(out/"matched_day_summary.csv",index=False)
    de.pivot_table(index=["gene_id","gene_name"],columns="day",
                   values="effect_mean_log2CPM_DOX_minus_UT").reset_index().to_csv(
                       out/"temporal_effects.csv",index=False)
    pd.DataFrame(cpm,index=df.gene_id,columns=samples).to_csv(out/"CPM.csv.gz",compression="gzip")
    pd.DataFrame(logcpm,index=df.gene_id,columns=samples).to_csv(
        out/"log2_CPM_plus_1.csv.gz",compression="gzip")
    print("\nGSE293056 complete")
    print(pd.DataFrame(summary))

# ---------------------------------------------------------------------
# GSE300568
# ---------------------------------------------------------------------
def analyze_gse300568(data_dir,outdir,max_cells=2500,seed=SEED):
    out=ensure_dir(outdir); rng=np.random.default_rng(seed)
    prefixes=discover_10x(data_dir)
    if not prefixes: raise RuntimeError("No GSE300568 10x matrices found.")

    qcs=[]; rna_pb={}; adt_pb={}; sampled=[]
    for prefix in prefixes:
        mat,feat,bc=read_10x(prefix)
        label=prefix.name
        m=re.search(r"Patient[_-]?(\d+)",label,re.I)
        patient=f"Patient{m.group(1)}" if m else label
        typ=feat.feature_type.astype(str)
        ri=np.where(typ.str.contains("Gene Expression",case=False,regex=False))[0]
        ai=np.where(typ.str.contains("Antibody Capture",case=False,regex=False))[0]
        R=mat[ri].T.tocsr()
        A=mat[ai].T.tocsr()
        genes=feat.iloc[ri].feature_name.astype(str).to_numpy()
        proteins=feat.iloc[ai].feature_name.astype(str).to_numpy()

        umi=np.asarray(R.sum(1)).ravel()
        detected=np.asarray((R>0).sum(1)).ravel()
        mito=np.char.startswith(np.char.upper(genes),"MT-")
        mt=np.asarray(R[:,mito].sum(1)).ravel() if mito.any() else np.zeros(len(umi))
        mtfrac=np.divide(mt,umi,out=np.zeros(len(umi),float),where=umi>0)
        qcs.append(pd.DataFrame({"patient":patient,"barcode":bc.values,
                                 "rna_umi":umi,"detected_genes":detected,
                                 "mitochondrial_fraction":mtfrac}))

        rna_pb[patient]=pd.Series(np.asarray(R.sum(0)).ravel(),
                                  index=genes).groupby(level=0).sum()
        if len(ai):
            adt_pb[patient]=pd.Series(np.asarray(A.sum(0)).ravel(),
                                      index=proteins).groupby(level=0).sum()

        n=min(max_cells,R.shape[0])
        ix=rng.choice(R.shape[0],n,replace=False)
        S=R[ix].astype(float)
        lib=np.asarray(S.sum(1)).ravel()
        sf=np.divide(1e4,lib,out=np.zeros(len(lib)),where=lib>0)
        S=sparse.diags(sf)@S
        S.data=np.log1p(S.data)
        sampled.append((patient,S))

    qc=pd.concat(qcs,ignore_index=True)
    qc.to_csv(out/"cell_qc.csv",index=False)
    qc.groupby("patient").agg(
        cells=("barcode","size"),
        median_rna_umi=("rna_umi","median"),
        median_detected_genes=("detected_genes","median"),
        median_mito_fraction=("mitochondrial_fraction","median")
    ).to_csv(out/"patient_qc_summary.csv")

    PB=pd.DataFrame(rna_pb).fillna(0)
    C=PB.divide(PB.sum(0),axis=1)*1e6
    keep=(C>1).sum(1)>=2
    L=np.log2(C.loc[keep]+1)
    model=PCA(2).fit(L.T); Z=model.transform(L.T)
    pd.DataFrame({"patient":L.columns,"PC1":Z[:,0],"PC2":Z[:,1],
                  "PC1_variance_percent":model.explained_variance_ratio_[0]*100,
                  "PC2_variance_percent":model.explained_variance_ratio_[1]*100
                  }).to_csv(out/"patient_pseudobulk_RNA_PCA.csv",index=False)

    top=L.var(1).sort_values(ascending=False).head(20).index
    z=L.sub(L.mean(1),axis=0).div(L.std(1).replace(0,np.nan),axis=0)
    z.loc[top].to_csv(out/"top20_variable_RNA_zscores.csv")

    if adt_pb:
        ADT=pd.DataFrame(adt_pb).fillna(0)
        ADT.divide(ADT.sum(0).replace(0,np.nan),axis=1).to_csv(
            out/"ADT_relative_abundance.csv")

    S=sparse.vstack([x for _,x in sampled])
    labels=np.concatenate([[p]*x.shape[0] for p,x in sampled])
    svd=TruncatedSVD(2,random_state=seed)
    Y=svd.fit_transform(S)
    pd.DataFrame({"Dim1":Y[:,0],"Dim2":Y[:,1],"patient":labels}).to_csv(
        out/"cell_TruncatedSVD_coordinates.csv",index=False)
    print("\nGSE300568 complete:",len(qc),"cells")

# ---------------------------------------------------------------------
# GSE310778
# ---------------------------------------------------------------------
def analyze_gse310778(matrix_prefix,metadata_file,outdir):
    out=ensure_dir(outdir)
    mat,feat,bc=read_10x(Path(matrix_prefix))
    md=pd.read_csv(metadata_file)
    required={"barcode","pair_id","perturbation"}
    if not required.issubset(md.columns):
        raise ValueError(f"GSE310778 metadata requires {sorted(required)}")

    pos=pd.Series(np.arange(len(bc)),index=bc.values)
    md=md[md.barcode.isin(pos.index)].copy()
    md["column"]=md.barcode.map(pos)
    groups=md.groupby(["pair_id","perturbation"]).column.apply(list)

    typ=feat.feature_type.astype(str)
    ri=np.where(typ.str.contains("Gene Expression",case=False,regex=False))[0]
    ai=np.where(typ.str.contains("Antibody Capture",case=False,regex=False))[0]

    def pseudobulk(feature_indices):
        return np.column_stack([
            np.asarray(mat[feature_indices][:,cols].sum(1)).ravel()
            for cols in groups])

    R=pseudobulk(ri)
    keys=list(groups.index)
    lib=R.sum(0); lib[lib==0]=1
    L=np.log2(R/lib*1e6+1)

    pca=PCA(n_components=min(5,len(keys))).fit(L.T)
    Z=pca.transform(L.T)
    pc=pd.DataFrame(Z,index=pd.MultiIndex.from_tuples(keys,names=["pair_id","perturbation"]))
    pc.to_csv(out/"RNA_pseudobulk_PCA.csv")
    pd.DataFrame({"PC":np.arange(1,len(pca.explained_variance_ratio_)+1),
                  "variance_ratio":pca.explained_variance_ratio_}).to_csv(
                      out/"RNA_pseudobulk_PCA_variance.csv",index=False)

    pairs=sorted(set(k[0] for k in keys))
    complete=[q for q in pairs if (q,"sgElovl1") in keys and (q,"sgRosa") in keys]
    ia=[keys.index((q,"sgElovl1")) for q in complete]
    ib=[keys.index((q,"sgRosa")) for q in complete]
    st,pv=ttest_rel(L[:,ia],L[:,ib],axis=1,nan_policy="omit")
    pv=np.nan_to_num(pv,nan=1.0)
    effect=(L[:,ia]-L[:,ib]).mean(1)
    tab=pd.DataFrame({
        "feature_id":feat.iloc[ri].feature_id.values,
        "gene_symbol":feat.iloc[ri].feature_name.values,
        "effect_mean_log2CPM_sgElovl1_minus_sgRosa":effect,
        "statistic":st,"pvalue":pv})
    tab["padj_bh"]=bh_fdr(pv)
    tab["passes_combined_threshold"]=(tab.padj_bh<0.05)&(
        tab.effect_mean_log2CPM_sgElovl1_minus_sgRosa.abs()>=1)
    tab.to_csv(out/"paired_RNA_statistics.csv",index=False)

    if len(ai):
        A=pseudobulk(ai)
        rel=A/np.where(A.sum(0)==0,1,A.sum(0))
        pd.DataFrame(rel,index=feat.iloc[ai].feature_name.values,
                     columns=pd.MultiIndex.from_tuples(keys)).to_csv(
                         out/"ADT_relative_abundance.csv")
        C=np.log1p(A)-np.log1p(A).mean(0,keepdims=True)
        st,pv=ttest_rel(C[:,ia],C[:,ib],axis=1,nan_policy="omit")
        at=pd.DataFrame({
            "feature":feat.iloc[ai].feature_name.values,
            "mean_paired_CLR_difference":(C[:,ia]-C[:,ib]).mean(1),
            "statistic":st,"pvalue":np.nan_to_num(pv,nan=1.0)})
        at["padj_bh"]=bh_fdr(at.pvalue)
        at.to_csv(out/"paired_ADT_statistics.csv",index=False)

    print("\nGSE310778 complete:",len(complete),"paired biological replicates")
    print("Genes passing combined RNA threshold:",int(tab.passes_combined_threshold.sum()))

# ---------------------------------------------------------------------
# Hallmark GSEA
# ---------------------------------------------------------------------
def hallmark_gsea(statistics_file,gmt,outdir,permutations=5000):
    try:
        import gseapy as gp
    except ImportError as e:
        raise ImportError("Install gseapy: pip install gseapy") from e

    out=ensure_dir(outdir)
    d=pd.read_csv(statistics_file).dropna(subset=["gene_symbol","statistic"])
    rank=d.groupby("gene_symbol").statistic.mean().sort_values(ascending=False)
    rank.to_csv(out/"ranked_gene_statistic.rnk",sep="\t",header=False)
    pre=gp.prerank(rnk=rank,gene_sets=gmt,permutation_num=permutations,
                   seed=SEED,min_size=10,max_size=500,outdir=str(out),verbose=True)
    pre.res2d.reset_index().to_csv(out/"Hallmark_GSEA_results.csv",index=False)
    print("\nHallmark GSEA complete:",len(pre.res2d),"gene sets")

# ---------------------------------------------------------------------
# Figures supported by completed result tables
# ---------------------------------------------------------------------
def generate_figures(results_dir,figures_dir):
    import matplotlib.pyplot as plt
    out=ensure_dir(figures_dir); r=Path(results_dir)

    # GSE293056 PCA
    p=r/"GSE293056/pca_scores.csv"
    if p.exists():
        d=pd.read_csv(p)
        fig,ax=plt.subplots(figsize=(8,7))
        for c,g in d.groupby("condition"):
            ax.scatter(g.PC1,g.PC2,label=c)
            for _,z in g.iterrows(): ax.annotate(str(z["day"]),(z.PC1,z.PC2),fontsize=7)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
        ax.set_title("GSE293056 sample structure")
        fig.tight_layout(); fig.savefig(out/"GSE293056_PCA.png",dpi=300); plt.close(fig)

    # GSE293056 matched-day effects
    p=r/"GSE293056/matched_day_welch_tests.csv"
    if p.exists():
        d=pd.read_csv(p); days=sorted(d.day.unique())
        fig,axs=plt.subplots(1,len(days),figsize=(4*len(days),4),sharey=True)
        axs=np.atleast_1d(axs)
        for ax,day in zip(axs,days):
            x=d[d.day==day]
            ax.scatter(x.effect_mean_log2CPM_DOX_minus_UT,
                       -np.log10(x.pvalue.clip(lower=1e-300)),s=5,alpha=.5)
            ax.axvline(-1,ls="--"); ax.axvline(1,ls="--")
            ax.set_title(f"Day {day}")
            ax.set_xlabel("Mean log2(CPM+1) difference")
        axs[0].set_ylabel("-log10 P")
        fig.tight_layout(); fig.savefig(out/"GSE293056_matched_day_effects.png",dpi=300); plt.close(fig)

    # GSE300568 patient PCA
    p=r/"GSE300568/patient_pseudobulk_RNA_PCA.csv"
    if p.exists():
        d=pd.read_csv(p); fig,ax=plt.subplots(figsize=(8,7))
        ax.scatter(d.PC1,d.PC2)
        for _,z in d.iterrows(): ax.annotate(str(z.patient),(z.PC1,z.PC2),fontsize=8)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("GSE300568 patient pseudobulk PCA")
        fig.tight_layout(); fig.savefig(out/"GSE300568_patient_PCA.png",dpi=300); plt.close(fig)

    # GSE300568 cell SVD
    p=r/"GSE300568/cell_TruncatedSVD_coordinates.csv"
    if p.exists():
        d=pd.read_csv(p); fig,ax=plt.subplots(figsize=(8,7))
        for patient,g in d.groupby("patient"): ax.scatter(g.Dim1,g.Dim2,s=2,label=patient)
        ax.set_xlabel("SVD1"); ax.set_ylabel("SVD2"); ax.set_title("GSE300568 cell-level TruncatedSVD")
        ax.legend(markerscale=4,fontsize=7)
        fig.tight_layout(); fig.savefig(out/"GSE300568_cell_TruncatedSVD.png",dpi=300); plt.close(fig)

    # GSE310778 RNA paired statistics
    p=r/"GSE310778/paired_RNA_statistics.csv"
    if p.exists():
        d=pd.read_csv(p); fig,ax=plt.subplots(figsize=(8,7))
        ax.scatter(d.effect_mean_log2CPM_sgElovl1_minus_sgRosa,
                   -np.log10(d.pvalue.clip(lower=1e-300)),s=7,alpha=.5)
        ax.axvline(-1,ls="--"); ax.axvline(1,ls="--")
        ax.axhline(-np.log10(.05),ls="--")
        ax.set_xlabel("Mean log2(CPM+1) difference: sgElovl1 - sgRosa")
        ax.set_ylabel("-log10 P"); ax.set_title("GSE310778 paired RNA response")
        fig.tight_layout(); fig.savefig(out/"GSE310778_paired_RNA.png",dpi=300); plt.close(fig)

    # GSEA
    p=r/"GSE310778/Hallmark_GSEA/Hallmark_GSEA_results.csv"
    if p.exists():
        d=pd.read_csv(p)
        nes=next((c for c in d.columns if c.upper()=="NES"),None)
        term=next((c for c in d.columns if c.lower() in ("term","name")),d.columns[0])
        if nes:
            x=d.reindex(d[nes].abs().sort_values(ascending=False).index).head(15).sort_values(nes)
            fig,ax=plt.subplots(figsize=(9,7)); ax.barh(x[term],x[nes])
            ax.set_xlabel("NES"); ax.set_title("Mouse Hallmark preranked GSEA")
            fig.tight_layout(); fig.savefig(out/"GSE310778_Hallmark_GSEA.png",dpi=300); plt.close(fig)

    print("\nSupported figures generated. No missing analysis was simulated.")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_parser():
    p=argparse.ArgumentParser(description="All-in-one AML multi-omics analysis")
    p.add_argument("command",choices=["all","gse293056","gse300568","gse310778","gsea","figures"])
    p.add_argument("--gse293056",default="data/GSE293056_salmon_gene_counts.csv.gz")
    p.add_argument("--gse300568",default="data/GSE300568")
    p.add_argument("--gse310778-prefix",default="data/GSE310778/GSE310778")
    p.add_argument("--gse310778-metadata",default="data/GSE310778/cell_metadata.csv")
    p.add_argument("--gmt",default="data/msigdb/mh.all.v2026.1.Mm.symbols.gmt")
    p.add_argument("--results",default="results")
    p.add_argument("--figures",default="figures")
    p.add_argument("--permutations",type=int,default=5000)
    p.add_argument("--max-cells-per-patient",type=int,default=2500)
    return p

def main():
    a=build_parser().parse_args()
    r=Path(a.results)
    if a.command in ("all","gse293056"):
        analyze_gse293056(a.gse293056,r/"GSE293056")
    if a.command in ("all","gse300568"):
        analyze_gse300568(a.gse300568,r/"GSE300568",a.max_cells_per_patient)
    if a.command in ("all","gse310778"):
        analyze_gse310778(a.gse310778_prefix,a.gse310778_metadata,r/"GSE310778")
    if a.command in ("all","gsea"):
        hallmark_gsea(r/"GSE310778/paired_RNA_statistics.csv",a.gmt,
                      r/"GSE310778/Hallmark_GSEA",a.permutations)
    if a.command in ("all","figures"):
        generate_figures(r,a.figures)

if __name__=="__main__":
    main()


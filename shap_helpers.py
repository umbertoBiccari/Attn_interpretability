import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path, PosixPath
import shap
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr



class TorchWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)

def save_fig(path_no_ext: Path):
    path_no_ext = Path(path_no_ext)
    plt.tight_layout()
    plt.savefig(str(path_no_ext) + ".png", dpi=250, bbox_inches="tight")
    plt.savefig(str(path_no_ext) + ".pdf", bbox_inches="tight")
    plt.close()

def _ensure_2d_shap(sv: np.ndarray, tag: str = "") -> np.ndarray:
    arr = np.asarray(sv)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"[{tag}] Expected 2D SHAP, got shape={arr.shape}")
    return arr.astype(float)

def mean_abs_shap(shap_vals_2d: np.ndarray, tag: str = "") -> np.ndarray:
    sv2 = _ensure_2d_shap(shap_vals_2d, tag=tag)
    return np.mean(np.abs(sv2), axis=0).astype(float).ravel()

def topk_table(mean_abs: np.ndarray, 
               names: list, 
               k: int,                
               tabdir: PosixPath) -> pd.DataFrame:
    ma = np.asarray(mean_abs, dtype=float).ravel()
    order = np.argsort(-ma).ravel()[:k].astype(int)
    df_top = pd.DataFrame({
        "rank": np.arange(1, len(order) + 1, dtype=int),
        "feature": [names[int(i)] for i in order],
        "mean_abs_shap": ma[order],
    })
    # df_top.to_csv(tabdir / f"mean_abs_shap_top{k}.csv", index=False, float_format="%.8f")
    return df_top

def global_table(mean_abs: np.ndarray, 
                 names: list,                  
                 tabdir: PosixPath) -> pd.DataFrame:
    ma = np.asarray(mean_abs, dtype=float).ravel()
    order = np.argsort(-ma).ravel().astype(int)
    df_g = pd.DataFrame({
        "rank": np.arange(1, len(order) + 1, dtype=int),
        "feature": [names[int(i)] for i in order],
        "mean_abs_shap": ma[order],
    })
    df_g.to_csv(tabdir / "mean_abs_shap_global.csv", index=False, float_format="%.8f")
    return df_g

def cosine_spearman(v1: np.ndarray, v2: np.ndarray):
    v1 = np.asarray(v1, dtype=float).ravel()
    v2 = np.asarray(v2, dtype=float).ravel()
    cs = float(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0])
    rho = float(spearmanr(v1, v2).correlation)
    return cs, rho

def compute_shap(model, model_tag: str, device, bg_t, ex_t):
    model.eval()
    wrapped = TorchWrapper(model).to(device)
    expl = shap.GradientExplainer(wrapped, bg_t)
    sv = expl.shap_values(ex_t)
    method = "GradientExplainer"
    
    if isinstance(sv, np.ndarray):
        sv_list = [sv[:, :, j] for j in range(sv.shape[2])]
    else:
        sv_list = list(sv)
        
    return sv_list, method

def compute_shap_3out(model, model_tag: str, device, bg_t, ex_t):
    model.eval()
    wrapped = TorchWrapper(model).to(device)
    expl = shap.GradientExplainer(wrapped, bg_t)
    sv = expl.shap_values(ex_t)
    method = "GradientExplainer"
    
    if isinstance(sv, np.ndarray):
        if sv.ndim != 3:
            raise ValueError(f"[{model_tag}] Unexpected SHAP array shape={sv.shape}")
        sv_list = [sv[:, :, j] for j in range(sv.shape[2])]
    else:
        sv_list = list(sv)

    if len(sv_list) != 3:
        raise ValueError(f"[{model_tag}] Expected 3 outputs, got {len(sv_list)}")
    return sv_list, method

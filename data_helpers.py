import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def encode_sex_to_01(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)

    ss = s.astype(str).str.strip().str.lower()
    mapping = {
        "male": 1, "m": 1, "man": 1, "1": 1, "true": 1,
        "female": 0, "f": 0, "woman": 0, "0": 0, "false": 0
    }
    out = ss.map(mapping)

    if out.isna().any():
        non_null = ss[ss.notna()]
        uniques = pd.Index(non_null.unique())
        if len(uniques) == 2:
            ordered = sorted(list(uniques))
            map2 = {ordered[0]: 0, ordered[1]: 1}
            out = ss.map(map2)
            print("[Info] Sex factorized using alphabetical order:", map2)
        else:
            bad_vals = ss[out.isna()].unique()[:10]
            raise ValueError(f"Could not encode sex. Example unrecognized values: {bad_vals}")

    return out.astype(float)

def filter_band(df_pairs, lo, hi, include_hi=False):
    if include_hi:
        mask = (df_pairs["abs_corr"] >= lo) & (df_pairs["abs_corr"] <= hi)
    else:
        mask = (df_pairs["abs_corr"] >= lo) & (df_pairs["abs_corr"] < hi)
    return df_pairs.loc[mask].copy()

def scale_support_table(df_raw: pd.DataFrame, cols: list) -> pd.DataFrame:
    X = df_raw[cols]
    out = pd.DataFrame(index=cols)
    out["missing_frac"] = X.isna().mean()
    out["zero_frac"] = (X.fillna(0) == 0).mean()
    out["n_unique"] = X.nunique(dropna=True)

    q25 = X.quantile(0.25)
    q75 = X.quantile(0.75)
    out["iqr"] = (q75 - q25).astype(float)
    out["std"] = X.std(skipna=True).astype(float)
    out["min"] = X.min(skipna=True).astype(float)
    out["max"] = X.max(skipna=True).astype(float)

    q01 = X.quantile(0.01)
    q99 = X.quantile(0.99)
    out["within_1_99_frac"] = ((X >= q01) & (X <= q99)).mean()

    return out.sort_values(["missing_frac", "zero_frac"], ascending=False)

def make_loader(Xn, yn, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(Xn),
        torch.from_numpy(yn),    # float32        
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def make_loader_multi(Xn, age, mets, sex, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(Xn),
        torch.from_numpy(age),    # float32        
        torch.from_numpy(mets),   # float32
        torch.from_numpy(sex),      # float32
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

# def make_loader_multi(Xn, age_s, agebin, mets_s, sex, batch_size, shuffle=True):
#     ds = TensorDataset(
#         torch.from_numpy(Xn),
#         torch.from_numpy(age_s),    # float32
#         torch.from_numpy(agebin),   # int64
#         torch.from_numpy(mets_s),   # float32
#         torch.from_numpy(sex),      # float32
#     )
#     return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
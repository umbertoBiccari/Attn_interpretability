from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, roc_auc_score, f1_score
)
import numpy as np
import torch

def inv_age(z): 
    return (z * age_std + age_mean).ravel()
    
def inv_mets(z): 
    return (z * mets_std + mets_mean).ravel()

def metrics_reg(y_true, y_pred):
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return r2, rmse, mae

def metrics_sex_acc_auc_f1(y_true01, logit, threshold=0.5):
    y = y_true01.astype(int).ravel()
    prob = 1.0 / (1.0 + np.exp(-logit.ravel()))
    pred = (prob >= threshold).astype(int)

    acc = float(accuracy_score(y, pred))
    try:
        auc = float(roc_auc_score(y, prob))
    except ValueError:
        auc = float("nan")

    if len(np.unique(y)) < 2:
        f1 = float("nan")
    else:
        f1 = float(f1_score(y, pred))
    return acc, auc, f1

@torch.no_grad()
def predict_np(model, Xn, device):
    model.eval()
    xb = torch.from_numpy(Xn).to(device)
    return model(xb).cpu().numpy()

@torch.no_grad()
def predict_np_single(model, Xn, device):
    model.eval()
    xb = torch.from_numpy(Xn).to(device)
    return model(xb).cpu().numpy()

def eval_metrics_from_out3(out3, age_true, mets_true, sex_true, sex_threshold=0.5):
    # age_mean, age_std = scalers["age_mean"], scalers["age_std"]
    # mets_mean, mets_std = scalers["mets_mean"], scalers["mets_std"]
    # age_pred = (out3[:, 0:1] * age_std + age_mean).ravel()
    # mets_pred = (out3[:, 1:2] * mets_std + mets_mean).ravel()
    age_pred = out3[:, 0:1].ravel()
    mets_pred = out3[:, 1:2].ravel()
    sex_log = out3[:, 2:3]

    r2a, rmsea, maea = metrics_reg(age_true.ravel(), age_pred)
    r2m, rmsem, maem = metrics_reg(mets_true.ravel(), mets_pred)
    accs, aucs, f1s = metrics_sex_acc_auc_f1(sex_true, sex_log, threshold=sex_threshold)

    return {
        "age_R2": r2a, "age_RMSE": rmsea, "age_MAE": maea,
        "MetSCORE_R2": r2m, "MetSCORE_RMSE": rmsem, "MetSCORE_MAE": maem,
        "sex_ACC": accs, "sex_AUC": aucs, "sex_F1": f1s,
    }

def eval_split_metrics_3out_direct(model3out, Xn, age_true, mets_true, sex_true, 
                                   device, sex_threshold=0.5):
    out3 = predict_np(model3out, Xn, device)
    return eval_metrics_from_out3(out3, age_true, mets_true, sex_true, 
                                  sex_threshold=sex_threshold)

def eval_single_metrics(model, Xn, yn, device):
    
    out = predict_np_single(model, Xn, device)
    r2, rmse, mae = metrics_reg(yn.ravel(), out)
    return {"R2": r2, "RMSE": rmse, "MAE": mae}    

# def eval_single_metrics_test(model, kind, Xn, yn, avg, std, device, sex_threshold=0.5):
#     out = predict_np_single(model, Xn, device)
#     if kind == "regression":
#         pred = (out * std + avg).ravel()
#         r2, rmse, mae = metrics_reg(yn.ravel(), pred)
#         return {"R2": r2, "RMSE": rmse, "MAE": mae, "ACC": np.nan, "AUC": np.nan, "F1": np.nan}    
#     if kind == "classification":
#         acc, auc, f1v = metrics_sex_acc_auc_f1(yn, out, threshold=sex_threshold)
#         return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "ACC": acc, "AUC": auc, "F1": f1v}
#     raise ValueError(kind)

def eval_single_metrics_test(model, Xn, yn, avg, std, device):
    
    out = predict_np_single(model, Xn, device)
    pred = (out * std + avg).ravel()
    r2, rmse, mae = metrics_reg(yn.ravel(), pred)
    
    return {"R2": r2, "RMSE": rmse, "MAE": mae}    
    

# def eval_single_metrics(model, kind, Xn, y_age_true, y_mets_true, y_sex_true, device, sex_threshold=0.5):
#     out = predict_np_single(model, Xn, device)
#     if kind == "age":
#         # pred = inv_age(out)
#         # r2, rmse, mae = metrics_reg(y_age_true.ravel(), pred)
#         r2, rmse, mae = metrics_reg(y_age_true.ravel(), out)
#         return {"R2": r2, "RMSE": rmse, "MAE": mae, "ACC": np.nan, "AUC": np.nan, "F1": np.nan}
#     if kind == "mets":
#         # pred = inv_mets(out)
#         # r2, rmse, mae = metrics_reg(y_mets_true.ravel(), pred)
#         r2, rmse, mae = metrics_reg(y_mets_true.ravel(), out)
#         return {"R2": r2, "RMSE": rmse, "MAE": mae, "ACC": np.nan, "AUC": np.nan, "F1": np.nan}
#     if kind == "sex":
#         acc, auc, f1v = metrics_sex_acc_auc_f1(y_sex_true, out, threshold=sex_threshold)
#         return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "ACC": acc, "AUC": auc, "F1": f1v}
#     raise ValueError(kind)
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

from metrics import eval_split_metrics_3out_direct, eval_single_metrics

import copy

def run_epoch_single(
    model,
    loader,
    device,
    loss_fn,
    optimizer=None,
    grad_clip_norm=None,
):
    train = optimizer is not None
    model.train(train)

    total, n = 0.0, 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        out = model(xb)
        loss = loss_fn(out, yb)

        if train:
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm
                )

            optimizer.step()

        bs = xb.size(0)
        total += loss.item() * bs
        n += bs

    return total / max(n, 1)


def train_single(
    model,
    loss_fn,
    optimizer,
    train_loader,
    val_loader,
    device,
    epochs=220,
    metrics_every=1,
    scheduler_patience=10,
    scheduler_factor=0.1,
    min_lr=1e-5,
    early_stopping_patience=6,
    min_delta=1e-6,
    grad_clip_norm=5.0,
):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )

    best_val = float("inf")
    best_state = None
    best_epoch = None
    bad_epochs = 0

    hist = []

    for ep in range(1, epochs + 1):

        start = time.time()
        
        tr_loss = run_epoch_single(
            model,
            train_loader,
            device,
            loss_fn,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
        )

        va_loss = run_epoch_single(
            model,
            val_loader,
            device,
            loss_fn,
            optimizer=None,
        )

        scheduler.step(va_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        improved = va_loss < best_val - min_delta

        if improved:
            best_val = va_loss
            best_epoch = ep
            bad_epochs = 0

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            bad_epochs += 1

        end = time.time()
        
        row = {
            "epoch": ep,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "lr": current_lr,
            "time": end-start
        }

        if ep % metrics_every == 0 or ep == 1:
            X_train = train_loader.dataset.tensors[0].detach().cpu().numpy()
            y_train = train_loader.dataset.tensors[1].detach().cpu().numpy()

            X_val = val_loader.dataset.tensors[0].detach().cpu().numpy()
            y_val = val_loader.dataset.tensors[1].detach().cpu().numpy()

            m_tr = eval_single_metrics(model, X_train, y_train, device)
            m_va = eval_single_metrics(model, X_val, y_val, device)

            row.update({
                "tr_R2": m_tr["R2"],
                "va_R2": m_va["R2"],
                "tr_RMSE": m_tr["RMSE"],
                "va_RMSE": m_va["RMSE"],
                "tr_MAE": m_tr["MAE"],
                "va_MAE": m_va["MAE"],
            })

            print(
                f"ep={ep:03d} "
                f"tr_loss={tr_loss:.4f} | va_loss={va_loss:.4f} | "
                f"tr R2={m_tr['R2']:.3f} | va R2={m_va['R2']:.3f} | "
                f"tr RMSE={m_tr['RMSE']:.3f} | va RMSE={m_va['RMSE']:.3f} | "
                f"tr MAE={m_tr['MAE']:.3f} | va MAE={m_va['MAE']:.3f} | "
                f"lr={current_lr:.2e} | bad_epochs={bad_epochs}"
            )

        hist.append(row)

        if bad_epochs >= early_stopping_patience:
            print(
                f"Early stopping at epoch {ep}. "
                f"Best epoch was {best_epoch} with val_loss={best_val:.6f}."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    hist_df = pd.DataFrame(hist)

    print(f"best epoch: {best_epoch}")
    print(f"best val loss: {best_val:.6f}")

    return hist_df
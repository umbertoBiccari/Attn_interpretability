import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_attention_rollout(
    attn_layers,
    add_residual=True,
    residual_weight=1.0,
    head_fusion="mean",
):
    """
    Compute attention rollout from a list of per-layer attention tensors.

    Parameters
    ----------
    attn_layers : list[torch.Tensor]
        List of attention tensors, one per transformer layer.
        Each tensor must have shape [B, H, N, N].
    add_residual : bool
        Whether to add identity before renormalization, as in attention rollout.
    residual_weight : float
        Weight for the identity matrix when add_residual=True.
    head_fusion : str
        How to fuse heads: "mean", "max", or "min".

    Returns
    -------
    rollout : torch.Tensor
        Tensor of shape [B, N, N], the cumulative attention rollout.
    """
    if len(attn_layers) == 0:
        raise ValueError("attn_layers is empty.")

    # Fuse heads in each layer
    fused = []
    for attn in attn_layers:
        # attn: [B, H, N, N]
        if attn.dim() != 4:
            raise ValueError(f"Expected attention tensor with shape [B, H, N, N], got {attn.shape}")

        if head_fusion == "mean":
            a = attn.mean(dim=1)   # [B, N, N]
        elif head_fusion == "max":
            a = attn.max(dim=1).values
        elif head_fusion == "min":
            a = attn.min(dim=1).values
        else:
            raise ValueError(f"Unsupported head_fusion='{head_fusion}'")

        fused.append(a)

    B, N, _ = fused[0].shape
    device = fused[0].device
    dtype = fused[0].dtype

    eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, N, N)

    # Add residual connection and renormalize row-wise
    processed = []
    for a in fused:
        if add_residual:
            a = a + residual_weight * eye
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-12)
        processed.append(a)

    # Rollout = A_L @ ... @ A_2 @ A_1
    rollout = processed[0]
    for a in processed[1:]:
        rollout = a.bmm(rollout)

    return rollout

@torch.no_grad()
def get_batch_rollout(
    model,
    x,
    device=None,
    add_residual=True,
    residual_weight=1.0,
    head_fusion="mean",
    return_outputs=True,
):
    """
    Compute attention rollout for one batch.

    Parameters
    ----------
    model : nn.Module
        Either TwoHeadTransformerNet or SingleOutTransformerNet.
    x : torch.Tensor
        Input batch of shape [B, d].
    device : torch.device or str or None
        Device on which to run the model.
    add_residual, residual_weight, head_fusion
        Passed to compute_attention_rollout.
    return_outputs : bool
        Whether to also return model outputs.

    Returns
    -------
    result : dict
        Keys:
            - "rollout": [B, N, N]
            - "cls_to_all": [B, N] if CLS token is used, else None
            - "feature_importance": [B, d]
            - "outputs": model predictions (optional)
            - "attn_layers": raw attention layers
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device
    x = x.to(device)

    outputs, attn_layers = model(x, return_attn=True)

    rollout = compute_attention_rollout(
        attn_layers,
        add_residual=add_residual,
        residual_weight=residual_weight,
        head_fusion=head_fusion,
    )  # [B, N, N]    
   
    use_cls = getattr(model.trunk, "use_cls_token", False)    

    if use_cls:
        # CLS attends to all tokens after rollout
        cls_to_all = rollout[:, 0, :]         # [B, N]
        feature_importance = cls_to_all[:, 1:]  # remove CLS itself -> [B, d]
        interactions = rollout[:, 1:, 1:]
    else:
        cls_to_all = None
        # No CLS token: average influence across all source tokens
        feature_importance = rollout.mean(dim=1)  # [B, N] where N=d
        interactions = rollout

    result = {
        "rollout": rollout,
        "cls_to_all": cls_to_all,
        "feature_importance": feature_importance,
        "interactions": interactions,
        "attn_layers": attn_layers,
    }

    if return_outputs:
        result["outputs"] = outputs

    return result

@torch.no_grad()
def compute_rollout_over_dataloader(
    model,
    dataloader,
    device=None,
    add_residual=True,
    residual_weight=1.0,
    head_fusion="mean",
    x_from_batch=None,
    y_from_batch=None,
):
    """
    Compute attention rollout over an entire test dataloader.

    Parameters
    ----------
    model : nn.Module
        Either TwoHeadTransformerNet or SingleOutTransformerNet.
    dataloader : DataLoader
        Test dataloader.
    device : torch.device or str or None
        Device for inference.
    add_residual, residual_weight, head_fusion
        Passed to compute_attention_rollout.
    x_from_batch : callable or None
        Function that extracts x from each batch.
        If None:
            - batch[0] is used for tuple/list batches
            - batch["x"] is used for dict batches
            - batch itself is used if it is a tensor
    y_from_batch : callable or None
        Optional function to extract labels/targets from batch.

    Returns
    -------
    results : dict
        Keys:
            - "outputs": [num_samples, out_dim]
            - "feature_importance": [num_samples, d]
            - "cls_to_all": [num_samples, N] or None
            - "targets": concatenated targets if available
            - "mean_feature_importance": [d]
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    all_outputs = []
    all_rollouts = []
    all_feature_importance = []
    all_interactions = []
    all_cls_to_all = []
    all_targets = []

    def default_x_from_batch(batch):
        if isinstance(batch, torch.Tensor):
            return batch
        if isinstance(batch, (tuple, list)):
            return batch[0]
        if isinstance(batch, dict):
            if "x" in batch:
                return batch["x"]
            raise KeyError("Batch is a dict but has no key 'x'. Please pass x_from_batch.")
        raise TypeError("Unsupported batch type. Please pass x_from_batch.")

    def default_y_from_batch(batch):
        if isinstance(batch, (tuple, list)) and len(batch) > 1:
            return batch[1]
        if isinstance(batch, dict):
            return batch.get("y", None)
        return None

    x_extractor = x_from_batch or default_x_from_batch
    y_extractor = y_from_batch or default_y_from_batch

    for batch in dataloader:        
        x = x_extractor(batch)
        y = y_extractor(batch)

        batch_result = get_batch_rollout(
            model=model,
            x=x,
            device=device,
            add_residual=add_residual,
            residual_weight=residual_weight,
            head_fusion=head_fusion,
            return_outputs=True,
        )

        all_outputs.append(batch_result["outputs"].detach().cpu())
        all_rollouts.append(batch_result["rollout"].detach().cpu())
        all_feature_importance.append(batch_result["feature_importance"].detach().cpu())
        all_interactions.append(batch_result["interactions"].detach().cpu())

        if batch_result["cls_to_all"] is not None:
            all_cls_to_all.append(batch_result["cls_to_all"].detach().cpu())

        if y is not None:
            all_targets.append(y.detach().cpu() if torch.is_tensor(y) else y)

    outputs = torch.cat(all_outputs, dim=0)
    feature_importance = torch.cat(all_feature_importance, dim=0)
    interactions = torch.cat(all_interactions, dim=0)
    cls_to_all = torch.cat(all_cls_to_all, dim=0) if len(all_cls_to_all) > 0 else None
    rollouts = torch.cat(all_rollouts, dim=0)

    results = {
        "outputs": outputs,
        "rollouts": rollouts,
        "feature_importance": feature_importance,
        "interactions": interactions,
        "cls_to_all": cls_to_all,  
    }

    if len(all_targets) > 0 and all(torch.is_tensor(t) for t in all_targets):
        results["targets"] = torch.cat(all_targets, dim=0)
    elif len(all_targets) > 0:
        results["targets"] = all_targets

    return results

import numpy as np
import matplotlib.pyplot as plt

def analyze_spectrum(
    R,
    sort_by="magnitude",
    plot_complex_plane=True,
    plot_singular_values=True,
    plot_eigenvalue_magnitudes=True,
    title_prefix="Rollout Matrix"
):
    """
    Analyze the spectrum of a rollout matrix.

    Parameters
    ----------
    R : np.ndarray
        Square rollout matrix of shape (n, n).
    sort_by : str
        How to sort eigenvalues for plotting magnitudes.
        Options:
            - "magnitude": sort by descending |lambda|
            - "real": sort by descending real part
    plot_complex_plane : bool
        Whether to scatter-plot eigenvalues in the complex plane.
    plot_singular_values : bool
        Whether to plot singular values.
    plot_eigenvalue_magnitudes : bool
        Whether to plot sorted eigenvalue magnitudes.
    title_prefix : str
        Prefix used in plot titles.

    Returns
    -------
    results : dict
        Dictionary with:
            - "eigenvalues"
            - "eigenvectors"
            - "singular_values"
            - "left_singular_vectors"
            - "right_singular_vectors"
            - "spectral_radius"
            - "effective_rank"
            - "trace"
            - "frobenius_norm"
    """
    R = np.asarray(R, dtype=np.float64)

    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square 2D matrix.")

    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eig(R)

    # SVD
    U, S, Vt = np.linalg.svd(R, full_matrices=False)

    # Basic summary quantities
    spectral_radius = np.max(np.abs(eigenvalues))
    trace = np.trace(R)
    frob_norm = np.linalg.norm(R, ord="fro")

    # Effective rank from singular values
    s_norm = S / np.sum(S) if np.sum(S) > 0 else S
    spectral_entropy = -np.sum(s_norm[s_norm > 0] * np.log(s_norm[s_norm > 0]))
    effective_rank = np.exp(spectral_entropy)

    # Sorting for plots
    if sort_by == "magnitude":
        idx = np.argsort(-np.abs(eigenvalues))
    elif sort_by == "real":
        idx = np.argsort(-np.real(eigenvalues))
    else:
        raise ValueError("sort_by must be 'magnitude' or 'real'.")

    eigenvalues_sorted = eigenvalues[idx]
    eig_mag_sorted = np.abs(eigenvalues_sorted)

    # ----- Plotting -----
    if plot_complex_plane:
        plt.figure(figsize=(6, 6))
        plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), alpha=0.8)
        theta = np.linspace(0, 2 * np.pi, 400)
        plt.plot(np.cos(theta), np.sin(theta), linestyle="--")  # unit circle
        plt.axhline(0)
        plt.axvline(0)
        plt.xlabel("Real part")
        plt.ylabel("Imaginary part")
        plt.title(f"{title_prefix}: Eigenvalues in Complex Plane")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.show()

    if plot_eigenvalue_magnitudes:
        plt.figure(figsize=(7, 4))
        plt.plot(eig_mag_sorted, marker="o")
        plt.xlabel("Index")
        plt.ylabel(r"$|\lambda_i|$")
        plt.title(f"{title_prefix}: Sorted Eigenvalue Magnitudes")
        plt.grid(True, alpha=0.3)
        plt.show()

    if plot_singular_values:
        plt.figure(figsize=(7, 4))
        plt.plot(S, marker="o")
        plt.xlabel("Index")
        plt.ylabel(r"$\sigma_i$")
        plt.title(f"{title_prefix}: Singular Values")
        plt.grid(True, alpha=0.3)
        plt.show()

        plt.figure(figsize=(7, 4))
        plt.semilogy(S, marker="o")
        plt.xlabel("Index")
        plt.ylabel(r"$\sigma_i$ (log scale)")
        plt.title(f"{title_prefix}: Singular Values (Log Scale)")
        plt.grid(True, alpha=0.3)
        plt.show()

    results = {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "singular_values": S,
        "left_singular_vectors": U,
        "right_singular_vectors": Vt.T,
        "spectral_radius": spectral_radius,
        "effective_rank": effective_rank,
        "trace": trace,
        "frobenius_norm": frob_norm,
    }

    return results

@torch.no_grad()
def extract_attention(
    model,
    dataloader,
    device=None
):
    """
    Extract attention matrices from a trained transformer given a dataloader.

    Parameters
    ----------
    model : nn.Module        
    dataloader : torch.DataLoader with N data divided in batches of size B.
    device : torch.device or str or None
        Device on which to run the model.    

    Returns
    -------
    attn_matrices : dict
        Keys: attention layer identifier
        Values: attention matrix for a given layer
                shape [N, d, d] with d shape of the input datum            
    """
    model.eval()

    attn_all = []
    
    for batch in dataloader:
        
        if isinstance(batch, torch.Tensor):
            x = batch
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        
        if device is None:
            device = next(model.parameters()).device
        x = x.to(device)    
        
        _, attn_layers = model(x, return_attn=True)

        attn_all.append(attn_layers)

    n_attn_layers = len(attn_all[0])
    
    result = list(map(list, zip(*attn_all)))
    attn_matrices = {}
    
    for layer_id, layer in enumerate(result):
        attn_layer = []
    
        for attn_matrix in layer:
            a = attn_matrix.mean(dim=1)
            attn_layer.append(a)

        attn_layer_torch = torch.cat(attn_layer, dim=0)
        attn_matrices[f"layer_{layer_id}"] = attn_layer_torch

    return attn_matrices
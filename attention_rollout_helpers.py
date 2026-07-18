import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_attention_rollout(
    attn_layers,
    add_residual=True,
    residual_weight=1.0,
    head_fusion="mean",
    normalize=True
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
        if normalize:            
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
    normalize=True
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

    outputs, attn_layers, _ = model(x, return_attn=True)

    rollout = compute_attention_rollout(
        attn_layers,
        add_residual=add_residual,
        residual_weight=residual_weight,
        head_fusion=head_fusion,
        normalize=normalize
    )  # [B, N, N]          

    result = {
        "rollout": rollout,
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
    normalize=True,
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
            normalize=normalize
        )

        all_outputs.append(batch_result["outputs"].detach().cpu())
        all_rollouts.append(batch_result["rollout"].detach().cpu())

        if y is not None:
            all_targets.append(y.detach().cpu() if torch.is_tensor(y) else y)

    outputs = torch.cat(all_outputs, dim=0)
    rollout = torch.cat(all_rollouts, dim=0)
    rollout_mean = rollout.mean(axis=0)

    results = {
        "outputs": outputs,
        "rollout": rollout,
        "rollout_mean": rollout_mean
    }

    if len(all_targets) > 0 and all(torch.is_tensor(t) for t in all_targets):
        results["targets"] = torch.cat(all_targets, dim=0)
    elif len(all_targets) > 0:
        results["targets"] = all_targets

    return results

def get_rollout_importance(rollout, alpha):

    rows_importance = rollout.sum(axis=1)
    columns_importance = rollout.sum(axis=0)

    global_importance = alpha*rows_importance + (1-alpha)*columns_importance

    results = {'rows': rows_importance, 
               'columns': columns_importance, 
               'global': global_importance}

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
    values_all = []
    
    for batch in dataloader:
        
        if isinstance(batch, torch.Tensor):
            x = batch
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        
        if device is None:
            device = next(model.parameters()).device
        x = x.to(device)    
        
        _, attn_layers, values = model(x, return_attn=True)

        attn_all.append(attn_layers)
        values_all.append(values)

    n_attn_layers = len(attn_all[0])    
    
    result_attn = list(map(list, zip(*attn_all)))
    result_values = list(map(list, zip(*values_all)))
    attn_matrices = {}
    values_matrices = {}
    
    for layer_id, layer in enumerate(result_attn):
        attn_layer = []
    
        for attn_matrix in layer:
            a = attn_matrix.mean(dim=1)
            attn_layer.append(a)

        attn_layer_torch = torch.cat(attn_layer, dim=0)
        attn_matrices[f"layer_{layer_id}"] = attn_layer_torch

    for layer_id, layer in enumerate(result_values):
        values_layer = []
    
        for matrix in layer:
            a = matrix.mean(dim=1)
            values_layer.append(a)            

        values_layer_torch = torch.cat(values_layer, dim=0)
        values_matrices[f"layer_{layer_id}"] = values_layer_torch

    return attn_matrices, values_matrices

def process_attention(attn_matrices):

    attn_matrices_dict = {}
    for layer in attn_matrices.keys():
        local_dict = {}
        attn_matrix = attn_matrices[layer].cpu().numpy()
        local_dict["attn_matrix"] = attn_matrix
        # local_dict["feature_importance"] = attn_matrix[:, 0, :]
        # local_dict["interactions"] = attn_matrix[:, 1:, 1:]

        attn_matrices_dict[layer] = local_dict    

    return attn_matrices_dict
"""Debug DConv weight loading."""

import numpy as np
import torch
import mlx.core as mx

from demucs.pretrained import get_model

from mlx_audio.models.demucs import HTDemucs


def check_dconv_weights():
    """Check if DConv weights are properly loaded."""
    print("\n=== DConv Weight Loading Check ===")

    # Load models
    pt_bag = get_model("htdemucs_ft")
    pt_model = pt_bag.models[0]
    pt_model.eval()

    mx_model = HTDemucs.from_pretrained("htdemucs_ft")

    # Get DConv from encoder[1]
    pt_dconv = pt_model.encoder[1].dconv
    mx_dconv = mx_model.encoder[1].dconv

    print("PyTorch DConv layers[0] weights:")
    for name, param in pt_dconv.layers[0].named_parameters():
        print(f"  {name}: shape={param.shape}, std={param.std().item():.6f}")

    print("\nMLX DConv layers[0] structure:")
    # Check if layers is registered properly
    print(f"  type(layers): {type(mx_dconv.layers)}")
    print(f"  len(layers): {len(mx_dconv.layers)}")

    # Try to access weights from first layer
    layer0 = mx_dconv.layers[0]
    print(f"  layer0 types: {[type(x).__name__ for x in layer0]}")

    # Check Conv1d weights
    conv1d_0 = layer0[0]  # First Conv1d
    print(f"\n  Conv1d[0] weight shape: {conv1d_0.weight.shape}")
    print(f"  Conv1d[0] weight std: {mx.std(conv1d_0.weight).item():.6f}")

    # Compare with PyTorch
    pt_conv1d_0 = pt_dconv.layers[0][0]
    print(f"\n  PyTorch Conv1d[0] weight shape: {pt_conv1d_0.weight.shape}")
    print(f"  PyTorch Conv1d[0] weight std: {pt_conv1d_0.weight.std().item():.6f}")

    # Check if weights match
    pt_w = pt_conv1d_0.weight.data.numpy()  # [out, in, k]
    mx_w = np.array(conv1d_0.weight)  # MLX: [out, k, in]

    # Reorder PyTorch to MLX format: [out, in, k] -> [out, k, in]
    pt_w_reordered = np.transpose(pt_w, (0, 2, 1))

    print(f"\n  PyTorch weight (reordered): shape={pt_w_reordered.shape}")
    print(f"  MLX weight: shape={mx_w.shape}")

    diff = np.abs(pt_w_reordered - mx_w)
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")

    if diff.max() > 1e-4:
        print("\n  WARNING: Weights differ!")
        print(f"  PT sample: {pt_w_reordered[0, 0, :3]}")
        print(f"  MLX sample: {mx_w[0, 0, :3]}")
    else:
        print("\n  Weights match!")


def check_layerscale_weights():
    """Check LayerScale weights specifically."""
    print("\n=== LayerScale Weight Check ===")

    pt_bag = get_model("htdemucs_ft")
    pt_model = pt_bag.models[0]
    mx_model = HTDemucs.from_pretrained("htdemucs_ft")

    pt_dconv = pt_model.encoder[1].dconv
    mx_dconv = mx_model.encoder[1].dconv

    # LayerScale is at index 6
    pt_ls = pt_dconv.layers[0][6]
    mx_ls = mx_dconv.layers[0][6]

    print(f"PyTorch LayerScale scale: shape={pt_ls.scale.shape}")
    print(f"  values[:5]: {pt_ls.scale[:5]}")
    print(f"  std: {pt_ls.scale.std().item():.6f}")

    print(f"\nMLX LayerScale scale: shape={mx_ls.scale.shape}")
    print(f"  values[:5]: {mx_ls.scale[:5]}")
    print(f"  std: {mx.std(mx_ls.scale).item():.6f}")

    # Compare
    pt_s = pt_ls.scale.data.numpy()
    mx_s = np.array(mx_ls.scale)

    diff = np.abs(pt_s - mx_s)
    print(f"\nMax diff: {diff.max():.6f}")

    if diff.max() > 1e-4:
        print("WARNING: LayerScale weights differ!")


def check_all_params():
    """List all parameters and check if DConv is included."""
    print("\n=== All Model Parameters Check ===")

    mx_model = HTDemucs.from_pretrained("htdemucs_ft")

    # Get all parameter names
    params = mx_model.parameters()

    def collect_param_names(params, prefix=""):
        names = []
        if isinstance(params, dict):
            for k, v in params.items():
                names.extend(collect_param_names(v, f"{prefix}.{k}" if prefix else k))
        elif isinstance(params, list):
            for i, v in enumerate(params):
                names.extend(collect_param_names(v, f"{prefix}.{i}"))
        elif isinstance(params, mx.array):
            names.append(prefix)
        return names

    param_names = collect_param_names(params)

    # Check for dconv parameters
    dconv_params = [n for n in param_names if "dconv" in n.lower()]
    print(f"Found {len(dconv_params)} dconv parameters:")
    for name in dconv_params[:20]:
        print(f"  {name}")
    if len(dconv_params) > 20:
        print(f"  ... and {len(dconv_params) - 20} more")


def test_dconv_forward():
    """Test DConv forward pass with matched weights."""
    print("\n=== DConv Forward Test ===")

    pt_bag = get_model("htdemucs_ft")
    pt_model = pt_bag.models[0]
    pt_model.eval()

    mx_model = HTDemucs.from_pretrained("htdemucs_ft")

    pt_dconv = pt_model.encoder[1].dconv
    mx_dconv = mx_model.encoder[1].dconv

    # Create test input
    np.random.seed(42)
    # Input shape for DConv: [B*Fr, T, C] where C=96 (chout of encoder[1])
    x_np = np.random.randn(128, 302, 96).astype(np.float32) * 0.07

    # PyTorch forward - need to be [B, C, T]
    with torch.no_grad():
        x_pt = torch.from_numpy(x_np.transpose(0, 2, 1))  # [B, T, C] -> [B, C, T]
        print(f"PyTorch input: shape={x_pt.shape}, std={x_pt.std().item():.6f}")
        out_pt = pt_dconv(x_pt)
        print(f"PyTorch output: shape={out_pt.shape}, std={out_pt.std().item():.6f}")

    # MLX forward - keep [B, T, C]
    x_mx = mx.array(x_np)  # [B, T, C]
    print(f"\nMLX input: shape={x_mx.shape}, std={mx.std(x_mx).item():.6f}")
    out_mx = mx_dconv(x_mx)
    mx.eval(out_mx)
    print(f"MLX output: shape={out_mx.shape}, std={mx.std(out_mx).item():.6f}")

    # Compare (transpose MLX output to match PyTorch)
    out_pt_np = out_pt.numpy().transpose(0, 2, 1)  # [B, C, T] -> [B, T, C]
    out_mx_np = np.array(out_mx)

    corr = np.corrcoef(out_pt_np.flatten(), out_mx_np.flatten())[0, 1]
    print(f"\nCorrelation: {corr:.6f}")
    print(f"PT/MLX std ratio: {out_pt.std().item() / mx.std(out_mx).item():.2f}")


if __name__ == "__main__":
    check_dconv_weights()
    check_layerscale_weights()
    check_all_params()
    test_dconv_forward()

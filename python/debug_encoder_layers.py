"""Debug encoder layers step by step."""

import numpy as np
import torch
import mlx.core as mx

from demucs.pretrained import get_model

from mlx_audio.models.demucs import HTDemucs


def compare_layer_by_layer():
    """Compare each encoder layer output."""
    print("\n=== Layer-by-Layer Encoder Comparison ===")

    # Load models
    pt_bag = get_model("htdemucs_ft")
    pt_model = pt_bag.models[0]
    pt_model.eval()

    mx_model = HTDemucs.from_pretrained("htdemucs_ft")

    # Create test audio
    np.random.seed(42)
    audio_np = np.random.randn(1, 2, 44100 * 7).astype(np.float32) * 0.1

    # Get PyTorch inputs
    with torch.no_grad():
        audio_pt = torch.from_numpy(audio_np)

        # Pad to training length
        training_length = int(pt_model.segment * pt_model.samplerate)
        if audio_pt.shape[-1] < training_length:
            audio_pt = torch.nn.functional.pad(
                audio_pt, (0, training_length - audio_pt.shape[-1])
            )

        z_pt = pt_model._spec(audio_pt)
        mag_pt = pt_model._magnitude(z_pt)

        mean_pt = mag_pt.mean(dim=(1, 2, 3), keepdim=True)
        std_pt = mag_pt.std(dim=(1, 2, 3), keepdim=True)
        x_pt = (mag_pt - mean_pt) / (1e-5 + std_pt)

        print(f"PyTorch input to encoder: shape={x_pt.shape}")
        print(f"  mean={x_pt.mean().item():.6f}, std={x_pt.std().item():.6f}")

        # Run through each encoder layer
        for idx, encode in enumerate(pt_model.encoder):
            x_pt = encode(x_pt, None)
            if idx == 0 and pt_model.freq_emb is not None:
                frs = torch.arange(x_pt.shape[-2], device=x_pt.device)
                emb = pt_model.freq_emb(frs).t()[None, :, :, None].expand_as(x_pt)
                x_pt = x_pt + pt_model.freq_emb_scale * emb

            print(f"  After encoder[{idx}]: shape={x_pt.shape}, "
                  f"mean={x_pt.mean().item():.6f}, std={x_pt.std().item():.6f}")

    # Get MLX inputs
    audio_mx = mx.array(audio_np)

    # Pad to training length
    B, C, T = audio_mx.shape
    training_length = int(mx_model.config.segment * mx_model.config.samplerate)
    if T < training_length:
        pad_amount = training_length - T
        mix = mx.pad(audio_mx, [(0, 0), (0, 0), (0, pad_amount)])
    else:
        mix = audio_mx

    # Compute STFT
    spec = mx_model._compute_stft(mix)

    # CAC conversion
    real_part = mx.real(spec)
    imag_part = mx.imag(spec)
    stacked = mx.stack([real_part, imag_part], axis=2)
    B_s, C_s, _, F_s, T_s = stacked.shape
    mag = stacked.reshape(B_s, C_s * 2, F_s, T_s)

    # Normalize
    spec_mean = mx.mean(mag, axis=(1, 2, 3), keepdims=True)
    spec_std = mx.std(mag, axis=(1, 2, 3), keepdims=True) + 1e-5
    freq_in = (mag - spec_mean) / spec_std

    print(f"\nMLX input to encoder: shape={freq_in.shape}")
    print(f"  mean={mx.mean(freq_in).item():.6f}, std={mx.std(freq_in).item():.6f}")

    # Convert to NHWC for MLX
    x_mx = freq_in.transpose(0, 2, 3, 1)

    # Run through each encoder layer
    for idx, enc in enumerate(mx_model.encoder):
        x_mx = enc(x_mx)
        if idx == 0 and hasattr(mx_model, "freq_emb"):
            frs = mx.arange(mx_model._n_freqs)
            emb = mx_model.freq_emb(frs)
            emb = emb[None, :, None, :]
            x_mx = x_mx + mx_model.config.freq_emb * emb
        mx.eval(x_mx)

        # Convert to NCHW for printing
        x_mx_nchw = x_mx.transpose(0, 3, 1, 2)
        print(f"  After encoder[{idx}]: shape={x_mx_nchw.shape}, "
              f"mean={mx.mean(x_mx_nchw).item():.6f}, std={mx.std(x_mx_nchw).item():.6f}")


def compare_first_encoder_conv():
    """Compare the first convolution in encoder[0]."""
    print("\n=== First Encoder Conv Comparison ===")

    # Load models
    pt_bag = get_model("htdemucs_ft")
    pt_model = pt_bag.models[0]
    pt_model.eval()

    mx_model = HTDemucs.from_pretrained("htdemucs_ft")

    # Get first encoder
    pt_enc = pt_model.encoder[0]
    mx_enc = mx_model.encoder[0]

    print(f"PyTorch encoder[0]: {pt_enc}")
    print(f"MLX encoder[0]: {mx_enc}")

    # Create simple test input
    np.random.seed(42)
    # PyTorch: [B, C, H, W] = [1, 4, 2048, 302]
    x_np = np.random.randn(1, 4, 2048, 302).astype(np.float32) * 0.1

    # PyTorch
    with torch.no_grad():
        x_pt = torch.from_numpy(x_np)
        # Get just the conv output
        # Looking at HEncLayer structure
        print(f"\nPyTorch encoder[0] structure:")
        for name, module in pt_enc.named_modules():
            if name:
                print(f"  {name}: {type(module).__name__}")

        # Run through encoder
        out_pt = pt_enc(x_pt, None)
        print(f"\nPyTorch encoder[0] output: shape={out_pt.shape}")
        print(f"  mean={out_pt.mean().item():.6f}, std={out_pt.std().item():.6f}")

    # MLX (input is NHWC)
    x_mx = mx.array(x_np)
    x_mx_nhwc = x_mx.transpose(0, 2, 3, 1)  # NCHW -> NHWC

    out_mx = mx_enc(x_mx_nhwc)
    mx.eval(out_mx)
    out_mx_nchw = out_mx.transpose(0, 3, 1, 2)  # NHWC -> NCHW

    print(f"\nMLX encoder[0] output: shape={out_mx_nchw.shape}")
    print(f"  mean={mx.mean(out_mx_nchw).item():.6f}, std={mx.std(out_mx_nchw).item():.6f}")

    # Compare weights
    print("\n=== Weight Comparison ===")

    # Get first conv weights from PyTorch
    # HEncLayer has conv structure
    print("\nPyTorch conv weight shapes:")
    for name, param in pt_enc.named_parameters():
        print(f"  {name}: {param.shape}")

    print("\nMLX conv weight shapes:")
    for name, param in mx.utils.tree_flatten(mx_enc.parameters()):
        if hasattr(param, 'shape'):
            print(f"  {name}: {param.shape}")


def compare_conv_weights_detailed():
    """Compare conv weights in detail."""
    print("\n=== Detailed Conv Weight Comparison ===")

    # Load models
    pt_bag = get_model("htdemucs_ft")
    pt_model = pt_bag.models[0]
    pt_model.eval()

    mx_model = HTDemucs.from_pretrained("htdemucs_ft")

    # Get first encoder conv weights
    # PyTorch HEncLayer.conv is Conv2d(4, 48, kernel_size=(8, 1), stride=(4, 1))

    # Find the conv in PyTorch
    pt_conv = pt_model.encoder[0].conv
    pt_weight = pt_conv.weight.data
    pt_bias = pt_conv.bias.data if pt_conv.bias is not None else None

    print(f"PyTorch conv weight: shape={pt_weight.shape}")
    print(f"  mean={pt_weight.mean().item():.6f}, std={pt_weight.std().item():.6f}")
    print(f"  min={pt_weight.min().item():.6f}, max={pt_weight.max().item():.6f}")
    if pt_bias is not None:
        print(f"PyTorch conv bias: shape={pt_bias.shape}")
        print(f"  mean={pt_bias.mean().item():.6f}, std={pt_bias.std().item():.6f}")

    # Find the conv in MLX
    mx_conv = mx_model.encoder[0].conv
    mx_weight = mx_conv.weight
    mx_bias = mx_conv.bias if hasattr(mx_conv, 'bias') else None

    print(f"\nMLX conv weight: shape={mx_weight.shape}")
    print(f"  mean={mx.mean(mx_weight).item():.6f}, std={mx.std(mx_weight).item():.6f}")
    print(f"  min={mx.min(mx_weight).item():.6f}, max={mx.max(mx_weight).item():.6f}")
    if mx_bias is not None:
        print(f"MLX conv bias: shape={mx_bias.shape}")
        print(f"  mean={mx.mean(mx_bias).item():.6f}, std={mx.std(mx_bias).item():.6f}")

    # Compare weights (need to handle different layouts)
    # PyTorch Conv2d weight: [out_channels, in_channels, kH, kW]
    # MLX Conv2d weight: [out_channels, kH, kW, in_channels] (OIHW vs OHWI)
    pt_w_np = pt_weight.numpy()  # [48, 4, 8, 1]
    mx_w_np = np.array(mx_weight)  # Should be [48, 8, 1, 4]

    print(f"\nLayout comparison:")
    print(f"  PyTorch weight layout: {pt_w_np.shape}")  # [O, I, H, W]
    print(f"  MLX weight layout: {mx_w_np.shape}")  # Should be [O, H, W, I]

    # Convert PyTorch layout to MLX layout for comparison
    pt_w_reordered = np.transpose(pt_w_np, (0, 2, 3, 1))  # [O, I, H, W] -> [O, H, W, I]
    print(f"  PyTorch weight reordered: {pt_w_reordered.shape}")

    diff = np.abs(pt_w_reordered - mx_w_np)
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")

    if diff.max() > 1e-4:
        print("\nWARNING: Weights differ significantly!")
        print(f"  PyTorch sample: {pt_w_reordered[0, 0, 0, :]}")
        print(f"  MLX sample: {mx_w_np[0, 0, 0, :]}")


if __name__ == "__main__":
    compare_layer_by_layer()
    compare_first_encoder_conv()
    compare_conv_weights_detailed()

"""Debug script to find the HTDemucs MLX vs PyTorch scaling discrepancy."""

import numpy as np
import torch
import mlx.core as mx

# Load PyTorch model
from demucs import pretrained
from demucs.apply import apply_model as pt_apply_model

# Load MLX model
from mlx_audio.models.demucs import HTDemucs

def compare_stft():
    """Compare STFT outputs."""
    print("\n=== STFT Comparison ===")

    # Create test audio
    np.random.seed(42)
    audio_np = np.random.randn(2, 44100 * 5).astype(np.float32) * 0.1

    # PyTorch STFT
    from demucs.spec import spectro
    audio_pt = torch.from_numpy(audio_np)
    z_pt = spectro(audio_pt, n_fft=4096, hop_length=1024)
    print(f"PyTorch STFT: shape={z_pt.shape}, abs.mean={z_pt.abs().mean().item():.6f}")

    # MLX STFT
    from mlx_audio.primitives import stft
    import math
    audio_mx = mx.array(audio_np)
    # Reshape for demucs convention
    audio_mx_flat = audio_mx.reshape(-1, audio_mx.shape[-1])  # [2, T]
    z_mx = stft(audio_mx_flat, n_fft=4096, hop_length=1024)
    # Apply normalization like demucs
    z_mx = z_mx / math.sqrt(4096)
    print(f"MLX STFT: shape={z_mx.shape}, abs.mean={mx.abs(z_mx).mean().item():.6f}")

    # Compare
    z_mx_np = np.array(z_mx)
    z_pt_np = z_pt.numpy()

    # Shapes might differ slightly in time dimension
    min_t = min(z_mx_np.shape[-1], z_pt_np.shape[-1])
    z_mx_trimmed = z_mx_np[..., :min_t]
    z_pt_trimmed = z_pt_np[..., :min_t]

    diff = np.abs(z_mx_trimmed - z_pt_trimmed)
    print(f"Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")

    corr = np.corrcoef(np.abs(z_mx_trimmed).flatten(), np.abs(z_pt_trimmed).flatten())[0, 1]
    print(f"Magnitude correlation: {corr:.6f}")

    return z_pt, z_mx


def compare_cac_conversion():
    """Compare complex-as-channels conversion."""
    print("\n=== CAC Conversion Comparison ===")

    # Create test STFT output
    np.random.seed(42)
    B, C, F, T = 1, 2, 2048, 216
    real = np.random.randn(B, C, F, T).astype(np.float32)
    imag = np.random.randn(B, C, F, T).astype(np.float32)
    z_np = real + 1j * imag

    # PyTorch CAC
    z_pt = torch.from_numpy(z_np)
    m_pt = torch.view_as_real(z_pt).permute(0, 1, 4, 2, 3)
    m_pt = m_pt.reshape(B, C * 2, F, T)
    print(f"PyTorch CAC: shape={m_pt.shape}")
    print(f"  Layout check - m_pt[0,0,0,:5] (should be real ch0): {m_pt[0,0,0,:5]}")
    print(f"  Layout check - m_pt[0,1,0,:5] (should be imag ch0): {m_pt[0,1,0,:5]}")

    # MLX CAC (matching model.py lines 219-225)
    z_mx = mx.array(z_np)
    real_part = mx.real(z_mx)  # [B, C, F, T]
    imag_part = mx.imag(z_mx)  # [B, C, F, T]
    stacked = mx.stack([real_part, imag_part], axis=2)  # [B, C, 2, F, T]
    B_s, C_s, _, F_s, T_s = stacked.shape
    m_mx = stacked.reshape(B_s, C_s * 2, F_s, T_s)
    print(f"MLX CAC: shape={m_mx.shape}")
    print(f"  Layout check - m_mx[0,0,0,:5] (should be real ch0): {m_mx[0,0,0,:5]}")
    print(f"  Layout check - m_mx[0,1,0,:5] (should be imag ch0): {m_mx[0,1,0,:5]}")

    # Verify layouts match
    m_pt_np = m_pt.numpy()
    m_mx_np = np.array(m_mx)

    diff = np.abs(m_pt_np - m_mx_np)
    print(f"Max diff: {diff.max():.6f}")

    if diff.max() > 1e-5:
        print("WARNING: CAC layout mismatch!")
        # Debug - check what each channel is
        for i in range(4):
            print(f"  Channel {i}: PyTorch={m_pt_np[0,i,0,0]:.4f}, MLX={m_mx_np[0,i,0,0]:.4f}")


def compare_normalization():
    """Compare input normalization."""
    print("\n=== Normalization Comparison ===")

    # Test data
    np.random.seed(42)
    x = np.random.randn(1, 4, 2048, 216).astype(np.float32)

    # PyTorch normalization
    x_pt = torch.from_numpy(x)
    mean_pt = x_pt.mean(dim=(1, 2, 3), keepdim=True)
    std_pt = x_pt.std(dim=(1, 2, 3), keepdim=True)
    x_norm_pt = (x_pt - mean_pt) / (1e-5 + std_pt)
    print(f"PyTorch: mean={mean_pt.item():.6f}, std={std_pt.item():.6f}")
    print(f"PyTorch normalized: mean={x_norm_pt.mean().item():.6f}, std={x_norm_pt.std().item():.6f}")

    # MLX normalization
    x_mx = mx.array(x)
    mean_mx = mx.mean(x_mx, axis=(1, 2, 3), keepdims=True)
    std_mx = mx.std(x_mx, axis=(1, 2, 3), keepdims=True) + 1e-5
    x_norm_mx = (x_mx - mean_mx) / std_mx
    print(f"MLX: mean={mean_mx.item():.6f}, std={std_mx.item():.6f}")
    print(f"MLX normalized: mean={mx.mean(x_norm_mx).item():.6f}, std={mx.std(x_norm_mx).item():.6f}")


def compare_encoder_output():
    """Compare encoder outputs."""
    print("\n=== Encoder Output Comparison ===")

    # Load models - use htdemucs (single model) instead of htdemucs_ft (bag)
    from demucs.pretrained import get_model
    pt_bag = get_model("htdemucs_ft")
    # Get first model from bag
    pt_model = pt_bag.models[0]
    pt_model.eval()

    mx_model = HTDemucs.from_pretrained("htdemucs_ft")

    # Create test audio
    np.random.seed(42)
    audio_np = np.random.randn(1, 2, 44100 * 7).astype(np.float32) * 0.1

    # Get PyTorch encoder output
    with torch.no_grad():
        audio_pt = torch.from_numpy(audio_np)
        z_pt = pt_model._spec(audio_pt)
        mag_pt = pt_model._magnitude(z_pt)

        mean_pt = mag_pt.mean(dim=(1, 2, 3), keepdim=True)
        std_pt = mag_pt.std(dim=(1, 2, 3), keepdim=True)
        x_pt = (mag_pt - mean_pt) / (1e-5 + std_pt)

        # Run through encoders
        saved_pt = []
        for idx, encode in enumerate(pt_model.encoder):
            x_pt = encode(x_pt, None)
            if idx == 0 and pt_model.freq_emb is not None:
                frs = torch.arange(x_pt.shape[-2], device=x_pt.device)
                emb = pt_model.freq_emb(frs).t()[None, :, :, None].expand_as(x_pt)
                x_pt = x_pt + pt_model.freq_emb_scale * emb
            saved_pt.append(x_pt.clone())

        print(f"PyTorch final encoder output: shape={x_pt.shape}")
        print(f"  mean={x_pt.mean().item():.6f}, std={x_pt.std().item():.6f}")

    # Get MLX encoder output
    audio_mx = mx.array(audio_np)

    # Manually run forward to get encoder output
    B, C, T = audio_mx.shape
    mix = audio_mx

    # Pad to training length
    training_length = int(mx_model.config.segment * mx_model.config.samplerate)
    if T < training_length:
        pad_amount = training_length - T
        mix = mx.pad(mix, [(0, 0), (0, 0), (0, pad_amount)])

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

    # Convert to NHWC
    x_mx = freq_in.transpose(0, 2, 3, 1)

    # Run through encoders
    for idx, enc in enumerate(mx_model.encoder):
        x_mx = enc(x_mx)
        if idx == 0 and hasattr(mx_model, "freq_emb"):
            frs = mx.arange(mx_model._n_freqs)
            emb = mx_model.freq_emb(frs)
            emb = emb[None, :, None, :]
            x_mx = x_mx + mx_model.config.freq_emb * emb

    # Convert back to NCHW for comparison
    x_mx_nchw = x_mx.transpose(0, 3, 1, 2)
    mx.eval(x_mx_nchw)

    print(f"MLX final encoder output: shape={x_mx_nchw.shape}")
    print(f"  mean={mx.mean(x_mx_nchw).item():.6f}, std={mx.std(x_mx_nchw).item():.6f}")

    # Compare
    x_pt_np = x_pt.numpy()
    x_mx_np = np.array(x_mx_nchw)

    min_shape = [min(a, b) for a, b in zip(x_pt_np.shape, x_mx_np.shape)]
    x_pt_trim = x_pt_np[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]
    x_mx_trim = x_mx_np[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]

    corr = np.corrcoef(x_pt_trim.flatten(), x_mx_trim.flatten())[0, 1]
    print(f"Correlation: {corr:.6f}")
    print(f"PT/MLX std ratio: {x_pt.std().item() / mx.std(x_mx_nchw).item():.2f}")


def compare_full_forward():
    """Compare full forward pass at key checkpoints."""
    print("\n=== Full Forward Comparison ===")

    # Load models - use first model from bag
    from demucs.pretrained import get_model
    pt_bag = get_model("htdemucs_ft")
    pt_model = pt_bag.models[0]
    pt_model.eval()

    mx_model = HTDemucs.from_pretrained("htdemucs_ft")

    # Create test audio
    np.random.seed(42)
    audio_np = np.random.randn(1, 2, 44100 * 7).astype(np.float32) * 0.1

    # PyTorch forward
    with torch.no_grad():
        audio_pt = torch.from_numpy(audio_np)
        output_pt = pt_model(audio_pt)
        print(f"PyTorch output: shape={output_pt.shape}")
        print(f"  mean={output_pt.mean().item():.6f}, std={output_pt.std().item():.6f}")
        for i, stem in enumerate(['drums', 'bass', 'other', 'vocals']):
            print(f"  {stem}: mean={output_pt[0,i].mean().item():.6f}, std={output_pt[0,i].std().item():.6f}")

    # MLX forward
    audio_mx = mx.array(audio_np)
    output_mx = mx_model(audio_mx)
    mx.eval(output_mx)

    print(f"\nMLX output: shape={output_mx.shape}")
    print(f"  mean={mx.mean(output_mx).item():.6f}, std={mx.std(output_mx).item():.6f}")
    for i, stem in enumerate(['drums', 'bass', 'other', 'vocals']):
        print(f"  {stem}: mean={mx.mean(output_mx[0,i]).item():.6f}, std={mx.std(output_mx[0,i]).item():.6f}")

    # Ratio
    pt_std = output_pt.std().item()
    mx_std = mx.std(output_mx).item()
    print(f"\nPT/MLX std ratio: {pt_std / mx_std:.2f}x")


if __name__ == "__main__":
    compare_stft()
    compare_cac_conversion()
    compare_normalization()
    compare_encoder_output()
    compare_full_forward()

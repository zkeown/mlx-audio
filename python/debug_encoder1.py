"""Debug encoder[1] in detail."""

import numpy as np
import torch
import mlx.core as mx

from demucs.pretrained import get_model

from mlx_audio.models.demucs import HTDemucs


def compare_encoder1_step_by_step():
    """Compare encoder[1] step by step."""
    print("\n=== Encoder[1] Step-by-Step Comparison ===")

    # Load models
    pt_bag = get_model("htdemucs_ft")
    pt_model = pt_bag.models[0]
    pt_model.eval()

    mx_model = HTDemucs.from_pretrained("htdemucs_ft")

    # Create input that matches encoder[0] output shape
    # PyTorch shape after encoder[0]: [1, 48, 512, 336]
    np.random.seed(42)
    x_np = np.random.randn(1, 48, 512, 302).astype(np.float32) * 0.3

    # Get PyTorch encoder[1]
    pt_enc = pt_model.encoder[1]
    print(f"PyTorch encoder[1] structure:")
    for name, module in pt_enc.named_modules():
        if name:
            print(f"  {name}: {type(module).__name__}")

    # Get MLX encoder[1]
    mx_enc = mx_model.encoder[1]

    # Run through encoder[1] step by step for PyTorch
    with torch.no_grad():
        x_pt = torch.from_numpy(x_np)
        print(f"\nPyTorch input: shape={x_pt.shape}, "
              f"mean={x_pt.mean().item():.6f}, std={x_pt.std().item():.6f}")

        # Step 1: conv
        y_pt = pt_enc.conv(x_pt)
        print(f"  After conv: shape={y_pt.shape}, "
              f"mean={y_pt.mean().item():.6f}, std={y_pt.std().item():.6f}")

        # Step 2: activation (likely GELU)
        if hasattr(pt_enc, 'act'):
            y_pt = pt_enc.act(y_pt)
            print(f"  After act: mean={y_pt.mean().item():.6f}, "
                  f"std={y_pt.std().item():.6f}")

        # Step 3: norm1
        y_pt = pt_enc.norm1(y_pt)
        print(f"  After norm1: mean={y_pt.mean().item():.6f}, "
              f"std={y_pt.std().item():.6f}")

        # Save for rewrite
        y_before_rewrite = y_pt.clone()

        # Step 4: dconv (if not empty)
        # PyTorch HEncLayer: x = y = self.norm1(self.act(self.conv(x)))
        # then dconv stuff...
        # Let's look at the exact forward
        print(f"\n  Checking dconv empty: {pt_enc.dconv}")

        # Actually run full encoder for accurate comparison
        out_pt = pt_enc(x_pt, None)
        print(f"\nPyTorch encoder[1] output: shape={out_pt.shape}, "
              f"mean={out_pt.mean().item():.6f}, std={out_pt.std().item():.6f}")

    # Run MLX encoder[1]
    x_mx = mx.array(x_np)
    x_mx_nhwc = x_mx.transpose(0, 2, 3, 1)  # NCHW -> NHWC

    print(f"\nMLX input (NHWC): shape={x_mx_nhwc.shape}, "
          f"mean={mx.mean(x_mx_nhwc).item():.6f}, std={mx.std(x_mx_nhwc).item():.6f}")

    out_mx = mx_enc(x_mx_nhwc)
    mx.eval(out_mx)
    out_mx_nchw = out_mx.transpose(0, 3, 1, 2)  # NHWC -> NCHW

    print(f"MLX encoder[1] output (NCHW): shape={out_mx_nchw.shape}, "
          f"mean={mx.mean(out_mx_nchw).item():.6f}, std={mx.std(out_mx_nchw).item():.6f}")

    print(f"\nPT/MLX std ratio: {out_pt.std().item() / mx.std(out_mx_nchw).item():.2f}")


def compare_henc_layer_impl():
    """Compare HEncLayer forward implementations."""
    print("\n=== HEncLayer Forward Comparison ===")

    # Load models
    pt_bag = get_model("htdemucs_ft")
    pt_model = pt_bag.models[0]
    pt_model.eval()

    mx_model = HTDemucs.from_pretrained("htdemucs_ft")

    # Use identical input
    np.random.seed(42)
    x_np = np.random.randn(1, 48, 512, 302).astype(np.float32) * 0.3

    pt_enc = pt_model.encoder[1]
    mx_enc = mx_model.encoder[1]

    print("Looking at PyTorch HEncLayer forward code...")
    import inspect
    # Get the forward method source
    try:
        src = inspect.getsource(pt_enc.forward)
        print(src[:2000])
    except Exception as e:
        print(f"Could not get source: {e}")

    # Now compare internal operations
    with torch.no_grad():
        x_pt = torch.from_numpy(x_np)

        # Manual step-through of HEncLayer forward
        # From demucs/hdemucs.py HEncLayer.forward:
        # x = y = self.norm1(self.act(self.conv(x)))
        # if self.dconv:
        #     y = self.dconv(y)
        # x = self.norm2(self.act(self.rewrite(y)))
        # if not self.empty:
        #     x = x + ... skip ...

        # Step 1: conv
        y1 = pt_enc.conv(x_pt)
        print(f"PT conv output: std={y1.std().item():.6f}")

        # Step 2: activation
        if hasattr(pt_enc, 'act'):
            y2 = pt_enc.act(y1)
            print(f"PT after act: std={y2.std().item():.6f}")
        else:
            y2 = y1
            print("PT no act")

        # Step 3: norm1
        y3 = pt_enc.norm1(y2)
        print(f"PT after norm1: std={y3.std().item():.6f}")

        # Step 4: dconv
        if pt_enc.dconv is not None:
            y4 = pt_enc.dconv(y3)
            print(f"PT after dconv: std={y4.std().item():.6f}")
        else:
            y4 = y3

        # Step 5: rewrite
        y5 = pt_enc.rewrite(y4)
        print(f"PT after rewrite: std={y5.std().item():.6f}")

        # Step 6: act again
        if hasattr(pt_enc, 'act'):
            y6 = pt_enc.act(y5)
            print(f"PT after act2: std={y6.std().item():.6f}")
        else:
            y6 = y5

        # Step 7: norm2
        y7 = pt_enc.norm2(y6)
        print(f"PT after norm2: std={y7.std().item():.6f}")

    # MLX equivalent
    x_mx = mx.array(x_np)
    x_mx_nhwc = x_mx.transpose(0, 2, 3, 1)

    # Manual step through MLX HEncLayer
    # Check MLX forward implementation
    print("\n--- MLX Steps ---")

    # Step 1: conv
    y1_mx = mx_enc.conv(x_mx_nhwc)
    mx.eval(y1_mx)
    print(f"MLX conv output: std={mx.std(y1_mx).item():.6f}")

    # Step 2: GELU activation
    y2_mx = mx.nn.gelu(y1_mx)
    mx.eval(y2_mx)
    print(f"MLX after gelu: std={mx.std(y2_mx).item():.6f}")

    # Step 3: norm1 (check if exists)
    if hasattr(mx_enc, 'norm1') and mx_enc.norm1 is not None:
        y3_mx = mx_enc.norm1(y2_mx)
        print(f"MLX after norm1: std={mx.std(y3_mx).item():.6f}")
    else:
        y3_mx = y2_mx
        print("MLX no norm1")

    # Step 4: dconv
    if hasattr(mx_enc, 'dconv') and mx_enc.dconv is not None:
        y4_mx = mx_enc.dconv(y3_mx)
        mx.eval(y4_mx)
        print(f"MLX after dconv: std={mx.std(y4_mx).item():.6f}")
    else:
        y4_mx = y3_mx

    # Step 5: rewrite
    y5_mx = mx_enc.rewrite(y4_mx)
    mx.eval(y5_mx)
    print(f"MLX after rewrite: std={mx.std(y5_mx).item():.6f}")

    # Step 6: GELU again
    y6_mx = mx.nn.gelu(y5_mx)
    mx.eval(y6_mx)
    print(f"MLX after gelu2: std={mx.std(y6_mx).item():.6f}")

    # Step 7: norm2
    if hasattr(mx_enc, 'norm2') and mx_enc.norm2 is not None:
        y7_mx = mx_enc.norm2(y6_mx)
        print(f"MLX after norm2: std={mx.std(y7_mx).item():.6f}")
    else:
        y7_mx = y6_mx
        print("MLX no norm2")


def compare_dconv():
    """Compare DConv operation in detail."""
    print("\n=== DConv Comparison ===")

    # Load models
    pt_bag = get_model("htdemucs_ft")
    pt_model = pt_bag.models[0]
    pt_model.eval()

    mx_model = HTDemucs.from_pretrained("htdemucs_ft")

    pt_dconv = pt_model.encoder[1].dconv
    mx_dconv = mx_model.encoder[1].dconv

    print(f"PyTorch DConv:\n{pt_dconv}")
    print(f"\nMLX DConv:\n{mx_dconv}")

    # Create test input that matches dconv input shape
    # After conv+act+norm: [B, 96, 128, T] for encoder[1]
    np.random.seed(42)
    x_np = np.random.randn(1, 96, 128, 302).astype(np.float32) * 0.07  # ~0.07 std

    with torch.no_grad():
        x_pt = torch.from_numpy(x_np)
        print(f"\nInput: shape={x_pt.shape}, std={x_pt.std().item():.6f}")

        out_pt = pt_dconv(x_pt)
        print(f"PyTorch DConv output: std={out_pt.std().item():.6f}")

    x_mx = mx.array(x_np)
    x_mx_nhwc = x_mx.transpose(0, 2, 3, 1)

    out_mx = mx_dconv(x_mx_nhwc)
    mx.eval(out_mx)
    out_mx_nchw = out_mx.transpose(0, 3, 1, 2)

    print(f"MLX DConv output: std={mx.std(out_mx_nchw).item():.6f}")

    print(f"\nPT/MLX std ratio: {out_pt.std().item() / mx.std(out_mx_nchw).item():.2f}")


if __name__ == "__main__":
    compare_encoder1_step_by_step()
    compare_henc_layer_impl()
    compare_dconv()

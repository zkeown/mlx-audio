"""Check what weight keys are in the safetensors file."""

import mlx.core as mx
from pathlib import Path


def check_safetensors_keys():
    """List all keys in the safetensors file."""
    print("\n=== Safetensors Keys ===")

    # Find the safetensors file
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    # Look for htdemucs_ft
    for model_dir in cache_dir.glob("models--lucasnewman--htdemucs*"):
        print(f"Found model dir: {model_dir}")
        for sf in model_dir.rglob("*.safetensors"):
            print(f"\n  File: {sf}")
            # Load and check keys
            weights = mx.load(str(sf))
            print(f"  Num keys: {len(weights)}")

            # Find dconv keys
            dconv_keys = [k for k in weights.keys() if "dconv" in k.lower()]
            print(f"\n  DConv keys ({len(dconv_keys)}):")
            for k in sorted(dconv_keys)[:30]:
                arr = weights[k]
                print(f"    {k}: shape={arr.shape}, std={mx.std(arr).item():.6f}")
            if len(dconv_keys) > 30:
                print(f"    ... and {len(dconv_keys) - 30} more")

            # Check a specific key
            test_key = "encoder.0.dconv.layers.0.0.weight"
            if test_key in weights:
                arr = weights[test_key]
                print(f"\n  {test_key}:")
                print(f"    shape={arr.shape}")
                print(f"    std={mx.std(arr).item():.6f}")
                print(f"    sample: {arr[0, 0, :3]}")
            else:
                print(f"\n  Key not found: {test_key}")
                # Find similar keys
                similar = [k for k in weights.keys() if "encoder.0.dconv" in k]
                print(f"  Similar keys: {similar[:10]}")


def check_model_param_names():
    """Check what the model expects."""
    from mlx_audio.models.demucs import HTDemucs

    print("\n=== Model Parameter Names ===")

    mx_model = HTDemucs.from_pretrained("htdemucs_ft")

    # Get trainable params
    params = mx_model.trainable_parameters()

    def collect_names(params, prefix=""):
        names = []
        if isinstance(params, dict):
            for k, v in params.items():
                names.extend(collect_names(v, f"{prefix}.{k}" if prefix else k))
        elif isinstance(params, list):
            for i, v in enumerate(params):
                names.extend(collect_names(v, f"{prefix}.{i}"))
        elif isinstance(params, mx.array):
            names.append(prefix)
        return names

    param_names = collect_names(params)

    dconv_names = [n for n in param_names if "dconv" in n.lower()]
    print(f"Model DConv params ({len(dconv_names)}):")
    for n in sorted(dconv_names)[:30]:
        print(f"  {n}")


if __name__ == "__main__":
    check_safetensors_keys()
    check_model_param_names()

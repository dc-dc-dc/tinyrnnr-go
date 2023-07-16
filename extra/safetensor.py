from safetensors import safe_open

with safe_open("./net.safetensors", framework="pt") as f:
    print(f.get_tensor("_conv_stem").flatten()[:10])
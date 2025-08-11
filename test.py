from safetensors.torch import load_file


file_path = "/workspace/fengzhuoer/andrew/outputs/ivypanda/abs2text/model.safetensors"
loaded = load_file(file_path, device="cuda:0")

print(loaded.keys())


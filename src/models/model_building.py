from pathlib import Path

curr_dir = Path(__file__)
home_dir = curr_dir.parent.parent.parent

params_path = home_dir.joinpath("params.yaml")


print(params_path)
print("======================================")
print(type(params_path))
#!/usr/bin/env python3
import os
import argparse
import re

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "config.py")

def update_config_file(workspace_dir=None, top_k=None, use_gpu=None):
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"config.py not found at {CONFIG_FILE}")

    # Define BASE_PATH automatically
    base_path = os.path.join(workspace_dir, "data", "PubMed") if workspace_dir else None

    with open(CONFIG_FILE, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # Update WORKSPACE_DIR
        if workspace_dir and re.match(r"^WORKSPACE_DIR\s*=", line):
            line = f'WORKSPACE_DIR = r"{workspace_dir}"\n'
        # Update BASE_PATH
        elif base_path and re.match(r"^BASE_PATH\s*=", line):
            line = f'BASE_PATH = r"{base_path}"\n'
        # Update TOP_K
        elif top_k and re.match(r"^TOP_K\s*=", line):
            line = f"TOP_K = {top_k}\n"
        # Update USE_GPU
        elif use_gpu is not None and re.match(r"^USE_GPU\s*=", line):
            line = f"USE_GPU = {use_gpu}\n"
        new_lines.append(line)

    # Write back updated config
    with open(CONFIG_FILE, "w") as f:
        f.writelines(new_lines)

    print("config.py updated successfully!")
    print(f"Workspace directory: {workspace_dir}")
    print(f"Base path: {base_path}")
    print(f"Top K: {top_k}, USE_GPU: {use_gpu}")


def main():
    parser = argparse.ArgumentParser(description="Update ragbio config.py permanently.")
    parser.add_argument("--workspace", type=str, required=True, help="Workspace directory")
    parser.add_argument("--top_k", type=int, default=10, help="Top K abstracts to retrieve")
    parser.add_argument("--use_gpu", type=str, choices=["true", "false"], default="true", help="Use GPU for embeddings")

    args = parser.parse_args()

    update_config_file(
        workspace_dir=args.workspace,
        top_k=args.top_k,
        use_gpu=True if args.use_gpu.lower() == "true" else False
    )


if __name__ == "__main__":
    main()

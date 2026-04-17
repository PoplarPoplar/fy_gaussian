from argparse import ArgumentParser
import subprocess
import os

def run_node_script(base_dir, aabbfile):
    command = ['node', 'index.js', base_dir, aabbfile]

    result = subprocess.run(command, capture_output=True, text=True)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("-b", "--base_dir", type=str, help="the lod result ply located in", required=True, default="")
    parser.add_argument("-a", "--aabbfile", type=str, help="the name of the ori_AABB_info.txt", required=True, default="")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.base_dir, args.aabbfile)):
        print(f"File {args.aabbfile} does not exist in {args.base_dir}.")
    else:
        run_node_script(args.base_dir, args.aabbfile)

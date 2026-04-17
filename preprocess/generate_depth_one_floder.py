import os, sys
import subprocess
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', default="", help="Will be set to project_dir/camera_calibration/rectified/images if not set")
    parser.add_argument('--chunks_dir', default="", help="Will be set to project_dir/camera_calibration/chunks if not set")
    parser.add_argument('--depth_generator', default="Depth-Anything-V2", choices=["DPT", "Depth-Anything-V2"], help="depth generator can be DPT or Depth-Anything-V2, we suggest using Depth-Anything-V2.")
    args = parser.parse_args()
    
    if args.images_dir == "":
        args.images_dir = os.path.join(args.project_dir, "camera_calibration/rectified/images")

    if args.chunks_dir == "":
        args.chunks_dir = os.path.join(args.project_dir, "camera_calibration/chunks")

    print(f"generating depth maps using {args.depth_generator}.")
    start_time = time.time()

    # Generate depth maps
    generator_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "submodules", args.depth_generator)
    
    if args.depth_generator == "DPT":
        base_generator_args = [
            "python", f"{generator_dir}/run_monodepth.py",
            "-t", "dpt_large"
        ]
    else:
        base_generator_args = [
            "python", f"{generator_dir}/run.py",
            "--encoder", "vitl", "--pred-only", "--grayscale"
        ]


    depth_folder = os.path.join(os.path.dirname(args.images_dir), "depths")
    os.makedirs(depth_folder, exist_ok=True)
    generator_args = base_generator_args + [
                    "--img-path", args.images_dir,
                    "--outdir", depth_folder
            ] 
    try:
        subprocess.run(generator_args, check=True, cwd=generator_dir)
    except subprocess.CalledProcessError as e:
        print(f"Error executing run_monodepth: {e}")
        sys.exit(1)


    # generate depth_params.json for each chunks
    print(f"generating depth_params.json for chunks {os.listdir(args.chunks_dir)}.")
    try:
        subprocess.run([
            "python", "preprocess/make_chunks_depth_scale.py", "--chunks_dir", f"{args.chunks_dir}", "--depths_dir", f"{depth_folder}"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing run_monodepth: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"Monocular depth estimation done in {(end_time - start_time)/60.0} minutes.")
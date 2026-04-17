import argparse
import os
import platform
import subprocess
import sys


def run_command(args):
    print("Running:", " ".join(args))
    subprocess.run(args, check=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setup_dirs(project_dir):
    camera_root = os.path.join(project_dir, "camera_calibration")
    ensure_dir(os.path.join(camera_root, "unrectified", "sparse"))
    ensure_dir(os.path.join(camera_root, "aligned", "sparse", "0"))
    ensure_dir(os.path.join(camera_root, "rectified"))
    return {
        "camera_root": camera_root,
        "database_path": os.path.join(camera_root, "unrectified", "database.db"),
        "unrectified_sparse_root": os.path.join(camera_root, "unrectified", "sparse"),
        "pose_prior_root": os.path.join(camera_root, "unrectified", "sparse_pose_prior"),
        "aligned_sparse_path": os.path.join(camera_root, "aligned", "sparse", "0"),
        "rectified_root": os.path.join(camera_root, "rectified"),
        "matching_path": os.path.join(camera_root, "unrectified", "matching.txt"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a rough Z-up COLMAP model from drone images using GPS pose priors."
    )
    parser.add_argument("--project_dir", required=True)
    parser.add_argument("--images_dir", default="", help="Defaults to <project_dir>/inputs/images")
    parser.add_argument(
        "--matcher",
        choices=["custom", "exhaustive"],
        default="custom",
        help="Use the repo's GPS-aware custom matcher or COLMAP exhaustive matcher.",
    )
    parser.add_argument("--camera_model", default="OPENCV")
    parser.add_argument("--default_focal_length_factor", default="0.5")
    parser.add_argument("--single_camera_per_folder", default="1")
    parser.add_argument("--n_gps_neighbours", type=int, default=25)
    parser.add_argument("--n_quad_matches_per_view", type=int, default=10)
    parser.add_argument("--alignment_type", default="enu", choices=["enu", "ecef", "enu-plane", "enu-plane-unscaled", "plane", "custom"])
    parser.add_argument("--alignment_max_error", type=float, default=20.0)
    parser.add_argument("--mapper_min_num_matches", type=int, default=15)
    parser.add_argument("--mapper_multiple_models", type=int, default=0)
    parser.add_argument("--max_image_size", type=int, default=4096)
    parser.add_argument("--skip_undistort", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.images_dir == "":
        args.images_dir = os.path.join(args.project_dir, "inputs", "images")

    if not os.path.isdir(args.images_dir):
        raise FileNotFoundError(f"Image directory does not exist: {args.images_dir}")

    colmap_exe = "colmap.bat" if platform.system() == "Windows" else "colmap"
    paths = setup_dirs(args.project_dir)

    run_command([
        colmap_exe,
        "feature_extractor",
        "--database_path",
        paths["database_path"],
        "--image_path",
        args.images_dir,
        "--ImageReader.single_camera_per_folder",
        str(args.single_camera_per_folder),
        "--ImageReader.default_focal_length_factor",
        str(args.default_focal_length_factor),
        "--ImageReader.camera_model",
        args.camera_model,
    ])

    if args.matcher == "custom":
        run_command([
            sys.executable,
            "preprocess/make_colmap_custom_matcher.py",
            "--image_path",
            args.images_dir,
            "--output_path",
            paths["matching_path"],
            "--n_gps_neighbours",
            str(args.n_gps_neighbours),
            "--n_quad_matches_per_view",
            str(args.n_quad_matches_per_view),
        ])
        run_command([
            colmap_exe,
            "matches_importer",
            "--database_path",
            paths["database_path"],
            "--match_list_path",
            paths["matching_path"],
        ])
    else:
        run_command([
            colmap_exe,
            "exhaustive_matcher",
            "--database_path",
            paths["database_path"],
        ])

    run_command([
        colmap_exe,
        "pose_prior_mapper",
        "--database_path",
        paths["database_path"],
        "--image_path",
        args.images_dir,
        "--output_path",
        paths["pose_prior_root"],
        "--Mapper.min_num_matches",
        str(args.mapper_min_num_matches),
        "--Mapper.multiple_models",
        str(args.mapper_multiple_models),
    ])

    pose_prior_model = os.path.join(paths["pose_prior_root"], "0")
    if not os.path.isdir(pose_prior_model):
        raise FileNotFoundError(f"pose_prior_mapper did not create expected model: {pose_prior_model}")

    run_command([
        colmap_exe,
        "model_aligner",
        "--input_path",
        pose_prior_model,
        "--output_path",
        paths["aligned_sparse_path"],
        "--database_path",
        paths["database_path"],
        "--ref_is_gps",
        "1",
        "--alignment_type",
        args.alignment_type,
        "--alignment_max_error",
        str(args.alignment_max_error),
    ])

    if not args.skip_undistort:
        run_command([
            colmap_exe,
            "image_undistorter",
            "--image_path",
            args.images_dir,
            "--input_path",
            paths["aligned_sparse_path"],
            "--output_path",
            paths["rectified_root"],
            "--output_type",
            "COLMAP",
            "--max_image_size",
            str(args.max_image_size),
        ])

    print("Z-up COLMAP pipeline completed.")
    print(f"Aligned sparse model: {paths['aligned_sparse_path']}")
    if not args.skip_undistort:
        print(f"Rectified workspace: {paths['rectified_root']}")


if __name__ == "__main__":
    main()

# 功能：封装无人机照片的最小 Z-up 空三流程，串联特征提取、GPS 邻接匹配、
# pose prior 建图、ENU 对齐和去畸变，并自动过滤非影像文件，为后续高斯流程提供方向正确的空三结果。

import argparse
import os
import platform
import subprocess
import sys

from exif import Image as ExifImage


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF")


def run_command(args):
    print("Running:", " ".join(args))
    subprocess.run(args, check=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def collect_image_relpaths(images_dir):
    image_relpaths = []
    for dirpath, dirnames, filenames in os.walk(images_dir):
        dirnames[:] = [name for name in dirnames if name != "camera_calibration"]
        rel_dir = os.path.relpath(dirpath, images_dir)
        rel_dir = "" if rel_dir == "." else rel_dir
        for filename in sorted(filenames):
            if filename.endswith(IMAGE_EXTENSIONS):
                if rel_dir:
                    image_relpaths.append(os.path.join(rel_dir, filename))
                else:
                    image_relpaths.append(filename)
    return image_relpaths


def dms_to_decimal(values, ref):
    decimal_value = values[0] + values[1] / 60.0 + values[2] / 3600.0
    if ref in ("S", "W"):
        decimal_value *= -1.0
    return decimal_value


def extract_gps_from_image(image_path):
    with open(image_path, "rb") as file:
        image = ExifImage(file)

    if not image.has_exif:
        return None

    required_attrs = [
        "gps_latitude",
        "gps_latitude_ref",
        "gps_longitude",
        "gps_longitude_ref",
        "gps_altitude",
    ]
    if not all(hasattr(image, attr) for attr in required_attrs):
        return None

    latitude = dms_to_decimal(image.gps_latitude, image.gps_latitude_ref)
    longitude = dms_to_decimal(image.gps_longitude, image.gps_longitude_ref)
    altitude = float(image.gps_altitude)
    return latitude, longitude, altitude


def write_ref_images_file(images_dir, image_relpaths, ref_images_path):
    ref_lines = []
    for image_relpath in image_relpaths:
        gps = extract_gps_from_image(os.path.join(images_dir, image_relpath))
        if gps is None:
            continue
        latitude, longitude, altitude = gps
        ref_lines.append(f"{image_relpath} {latitude:.12f} {longitude:.12f} {altitude:.6f}")

    if not ref_lines:
        raise RuntimeError("No usable EXIF GPS records were found for model alignment.")

    with open(ref_images_path, "w", encoding="utf-8") as file:
        file.write("\n".join(ref_lines))
        file.write("\n")


def setup_dirs(project_dir):
    camera_root = os.path.join(project_dir, "camera_calibration")
    ensure_dir(os.path.join(camera_root, "unrectified"))
    ensure_dir(os.path.join(camera_root, "unrectified", "sparse"))
    ensure_dir(os.path.join(camera_root, "unrectified", "sparse_pose_prior"))
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
        "image_list_path": os.path.join(camera_root, "unrectified", "image_list.txt"),
        "ref_images_path": os.path.join(camera_root, "unrectified", "ref_images.txt"),
    }


def default_workspace_dir(images_dir):
    images_dir = os.path.abspath(images_dir)
    parent_dir = os.path.dirname(images_dir)
    folder_name = os.path.basename(os.path.normpath(images_dir))
    return os.path.join(parent_dir, f"{folder_name}_camera_calibration")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a rough Z-up COLMAP model from drone images using GPS pose priors."
    )
    parser.add_argument("--project_dir", required=True)
    parser.add_argument("--images_dir", default="", help="Defaults to <project_dir>/inputs/images")
    parser.add_argument(
        "--workspace_dir",
        default="",
        help="Directory for generated outputs. Defaults to a sibling folder next to images_dir.",
    )
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
    parser.add_argument(
        "--alignment_type",
        default="enu",
        choices=["enu", "ecef", "enu-plane", "enu-plane-unscaled", "plane", "custom"],
    )
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
    if args.workspace_dir == "":
        if os.path.abspath(args.images_dir) == os.path.abspath(args.project_dir):
            args.workspace_dir = default_workspace_dir(args.images_dir)
        else:
            args.workspace_dir = args.project_dir

    if not os.path.isdir(args.images_dir):
        raise FileNotFoundError(f"Image directory does not exist: {args.images_dir}")

    colmap_exe = "colmap.bat" if platform.system() == "Windows" else "colmap"
    paths = setup_dirs(args.workspace_dir)
    image_relpaths = collect_image_relpaths(args.images_dir)
    if not image_relpaths:
        raise RuntimeError(f"No supported images were found under: {args.images_dir}")

    with open(paths["image_list_path"], "w", encoding="utf-8") as file:
        file.write("\n".join(image_relpaths))
        file.write("\n")
    write_ref_images_file(args.images_dir, image_relpaths, paths["ref_images_path"])

    run_command([
        colmap_exe,
        "feature_extractor",
        "--database_path",
        paths["database_path"],
        "--image_path",
        args.images_dir,
        "--image_list_path",
        paths["image_list_path"],
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
        "--Mapper.image_list_path",
        paths["image_list_path"],
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
        "--ref_images_path",
        paths["ref_images_path"],
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
    print(f"Workspace root: {paths['camera_root']}")


if __name__ == "__main__":
    main()

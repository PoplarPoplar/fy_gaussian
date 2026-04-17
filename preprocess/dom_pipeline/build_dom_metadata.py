# 功能：为单块或最小高斯结果生成 DOM 渲染所需的最小工程元数据，主要输出 BoundingBox.json 和 at.xml。

import argparse
import json
import os
from pathlib import Path

import numpy as np
from plyfile import PlyData


def read_bounds_file(bounds_dir: Path):
    center_path = bounds_dir / "center.txt"
    extent_path = bounds_dir / "extent.txt"
    if not center_path.exists() or not extent_path.exists():
        return None

    center = np.fromstring(center_path.read_text(encoding="utf-8").strip(), sep=" ")
    extent = np.fromstring(extent_path.read_text(encoding="utf-8").strip(), sep=" ")
    if center.size != 3 or extent.size != 3:
        raise ValueError(f"Invalid bounds file content in {bounds_dir}")
    return center, extent


def read_ply_bbox(ply_path: Path):
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    return xyz.min(axis=0), xyz.max(axis=0)


def build_bbox_json(origin, bbox_min, bbox_max, epsg):
    return {
        "Normalizer": {
            "origin": {
                "x": float(origin[0]),
                "y": float(origin[1]),
                "z": float(origin[2]),
            },
            "transform": {
                "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
            },
        },
        "Quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        "CurrentBoundingBox": {
            "min": {"x": float(bbox_min[0]), "y": float(bbox_min[1]), "z": float(bbox_min[2])},
            "max": {"x": float(bbox_max[0]), "y": float(bbox_max[1]), "z": float(bbox_max[2])},
        },
        "CoordinateSystem": {
            "output": epsg,
        },
    }


def write_coordinate_system_json(project_dir: Path, epsg: str):
    coord_json = {
        "input": {"type": 0, "value": "4326 WGS 84"},
        "output": {"type": 0, "value": epsg},
    }
    (project_dir / "CoordinateSystem.json").write_text(
        json.dumps(coord_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Build minimal DOM project metadata")
    parser.add_argument("--project_dir", required=True, help="DOM project directory")
    parser.add_argument("--ply_path", required=True, help="Gaussian point cloud ply path")
    parser.add_argument("--bounds_dir", default="", help="Optional chunk directory containing center.txt/extent.txt")
    parser.add_argument("--origin", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Normalizer origin xyz")
    parser.add_argument("--gsd", type=float, default=0.03, help="GSD written into at.xml")
    parser.add_argument("--epsg", default="4547 WGS 84", help="Output coordinate system description")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    ply_path = Path(args.ply_path)
    project_dir.mkdir(parents=True, exist_ok=True)

    ply_min, ply_max = read_ply_bbox(ply_path)
    bbox_min, bbox_max = ply_min, ply_max

    if args.bounds_dir:
        bounds = read_bounds_file(Path(args.bounds_dir))
        if bounds is not None:
            center, extent = bounds
            bbox_min = center - extent / 2
            bbox_max = center + extent / 2

    bbox_json = build_bbox_json(np.array(args.origin), bbox_min, bbox_max, args.epsg)
    (project_dir / "BoundingBox.json").write_text(
        json.dumps(bbox_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (project_dir / "at.xml").write_text(f"<DATA> <GSD>{args.gsd}</GSD> </DATA>", encoding="utf-8")
    write_coordinate_system_json(project_dir, args.epsg)
    print(f"BoundingBox.json -> {project_dir / 'BoundingBox.json'}")
    print(f"at.xml -> {project_dir / 'at.xml'}")
    print(f"CoordinateSystem.json -> {project_dir / 'CoordinateSystem.json'}")


if __name__ == "__main__":
    main()

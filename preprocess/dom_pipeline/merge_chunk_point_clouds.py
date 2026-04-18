#!/usr/bin/env python3
# 功能：将多个 chunk 的最终高斯点云 PLY 合并为一个总 PLY，供当前 DOM/TDOM 流程直接使用。

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def resolve_chunk_names(chunks_root: Path, chunk_names_arg: str) -> list[str]:
    if chunk_names_arg:
        return [name.strip() for name in chunk_names_arg.split(",") if name.strip()]

    valid_chunks_file = chunks_root / "valid_chunks.txt"
    if valid_chunks_file.exists():
        return [line.strip() for line in valid_chunks_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    return sorted([path.name for path in chunks_root.iterdir() if path.is_dir()])


def locate_chunk_ply(outputs_root: Path, chunk_name: str) -> Path:
    direct_output = outputs_root / chunk_name
    if direct_output.exists():
        candidate = direct_output / "point_cloud" / "iteration_final" / "point_cloud.ply"
        if candidate.exists():
            return candidate

    for path in sorted(outputs_root.glob(f"{chunk_name}*")):
        candidate = path / "point_cloud" / "iteration_final" / "point_cloud.ply"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"未找到 chunk {chunk_name} 的最终 point_cloud.ply")


def merge_chunk_point_clouds(outputs_root: Path, chunk_names: list[str], output_ply: Path) -> None:
    merged_vertices = []
    vertex_dtype = None

    for chunk_name in chunk_names:
        ply_path = locate_chunk_ply(outputs_root, chunk_name)
        ply_data = PlyData.read(str(ply_path))
        vertices = np.array(ply_data["vertex"].data)

        if vertex_dtype is None:
            vertex_dtype = vertices.dtype
        elif vertices.dtype != vertex_dtype:
            raise ValueError(f"chunk {chunk_name} 的顶点属性结构与前面不一致，不能直接拼接")

        merged_vertices.append(vertices)
        print(f"[merge] {chunk_name}: {len(vertices)} points <- {ply_path}")

    if not merged_vertices:
        raise ValueError("没有可合并的 chunk 点云")

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    merged_array = np.concatenate(merged_vertices, axis=0)
    merged_ply = PlyData([PlyElement.describe(merged_array, "vertex")], text=False)
    merged_ply.write(str(output_ply))
    print(f"[done] merged {len(chunk_names)} chunks, total points: {len(merged_array)}")
    print(f"[done] output ply: {output_ply}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并多个 chunk 的最终高斯点云 PLY")
    parser.add_argument("--chunks_root", required=True, help="chunk 目录，支持读取 valid_chunks.txt")
    parser.add_argument("--outputs_root", required=True, help="训练输出目录根路径")
    parser.add_argument("--output_ply", required=True, help="输出总 PLY 路径")
    parser.add_argument(
        "--chunk_names",
        default="",
        help="可选，逗号分隔的 chunk 名称列表；为空时优先读取 valid_chunks.txt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    chunks_root = Path(args.chunks_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()
    output_ply = Path(args.output_ply).resolve()

    chunk_names = resolve_chunk_names(chunks_root, args.chunk_names)
    print(f"[info] chunks: {chunk_names}")
    merge_chunk_point_clouds(outputs_root, chunk_names, output_ply)

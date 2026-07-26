from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bmesh
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_blend import _mesh_topology


def _mesh_object() -> bpy.types.Object:
    objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if len(objects) != 1:
        raise RuntimeError(f"Expected one mesh object, found {len(objects)}")
    return objects[0]


def analyze(distances: list[float]) -> dict[str, object]:
    source = _mesh_object()
    results = []
    for distance in distances:
        mesh = source.data.copy()
        obj = source.copy()
        obj.data = mesh
        before_vertices = len(mesh.vertices)

        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=distance)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        results.append(
            {
                "distance": distance,
                "vertices_before": before_vertices,
                "vertices_after": len(mesh.vertices),
                "vertices_merged": before_vertices - len(mesh.vertices),
                "edges": len(mesh.edges),
                "faces": len(mesh.polygons),
                "topology": _mesh_topology(obj),
            }
        )
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)

    return {
        "filepath": bpy.data.filepath,
        "object": source.name,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--distances",
        nargs="+",
        type=float,
        default=[1e-8, 1e-7, 1e-6, 1e-5],
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.write_text(
        json.dumps(analyze(args.distances), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

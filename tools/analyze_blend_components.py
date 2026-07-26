from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


def _single_mesh_object() -> bpy.types.Object:
    objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if len(objects) != 1:
        raise RuntimeError(f"Expected one mesh object, found {len(objects)}")
    return objects[0]


def analyze() -> dict[str, object]:
    obj = _single_mesh_object()
    mesh = obj.data
    parent = list(range(len(mesh.vertices)))
    ranks = bytearray(len(mesh.vertices))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if ranks[root_a] < ranks[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        if ranks[root_a] == ranks[root_b]:
            ranks[root_a] += 1

    for edge in mesh.edges:
        union(edge.vertices[0], edge.vertices[1])

    components: dict[int, dict[str, object]] = {}
    matrix = obj.matrix_world
    for vertex in mesh.vertices:
        root = find(vertex.index)
        point = matrix @ vertex.co
        component = components.setdefault(
            root,
            {
                "vertices": 0,
                "faces": 0,
                "triangles": 0,
                "bounds_min": [math.inf, math.inf, math.inf],
                "bounds_max": [-math.inf, -math.inf, -math.inf],
                "signed_volume": 0.0,
                "surface_area": 0.0,
            },
        )
        component["vertices"] += 1
        for axis in range(3):
            component["bounds_min"][axis] = min(
                component["bounds_min"][axis], point[axis]
            )
            component["bounds_max"][axis] = max(
                component["bounds_max"][axis], point[axis]
            )

    for polygon in mesh.polygons:
        vertices = list(polygon.vertices)
        root = find(vertices[0])
        component = components[root]
        component["faces"] += 1
        component["triangles"] += max(len(vertices) - 2, 0)
        p0 = matrix @ mesh.vertices[vertices[0]].co
        for index in range(1, len(vertices) - 1):
            p1 = matrix @ mesh.vertices[vertices[index]].co
            p2 = matrix @ mesh.vertices[vertices[index + 1]].co
            cross = (p1 - p0).cross(p2 - p0)
            component["surface_area"] += cross.length * 0.5
            component["signed_volume"] += p0.dot(p1.cross(p2)) / 6.0

    result = []
    for component in components.values():
        minimum = component.pop("bounds_min")
        maximum = component.pop("bounds_max")
        component["bounds"] = {
            "min": minimum,
            "max": maximum,
            "size": [maximum[i] - minimum[i] for i in range(3)],
            "center": [(minimum[i] + maximum[i]) * 0.5 for i in range(3)],
        }
        component["outward_winding"] = component["signed_volume"] > 0.0
        result.append(component)
    result.sort(key=lambda item: item["vertices"], reverse=True)

    return {
        "filepath": bpy.data.filepath,
        "object": obj.name,
        "object_scale": list(obj.scale),
        "components": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.write_text(
        json.dumps(analyze(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

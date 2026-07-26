from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdSkel


def _value(attribute: Usd.Attribute, default: Any = None) -> Any:
    if not attribute:
        return default
    value = attribute.Get(Usd.TimeCode.Default())
    return default if value is None else value


def _asset_path(value: Any) -> str | None:
    if isinstance(value, Sdf.AssetPath):
        return value.resolvedPath or value.path
    return None


def _bound_material(prim: Usd.Prim) -> str | None:
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    return str(material.GetPath()) if material else None


def _world_position(prim: Usd.Prim) -> list[float]:
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    return [float(value) for value in translation]


def _world_bounds(prim: Usd.Prim) -> dict[str, list[float]]:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    minimum = bounds.GetMin()
    maximum = bounds.GetMax()
    return {
        "min": [float(value) for value in minimum],
        "max": [float(value) for value in maximum],
        "size": [float(maximum[i] - minimum[i]) for i in range(3)],
    }


def _topology(points: Any, counts: Any, indices: Any) -> dict[str, Any]:
    parent = list(range(len(points)))
    ranks = [0] * len(points)

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

    edges: Counter[tuple[int, int]] = Counter()
    edge_orientations: Counter[tuple[int, int]] = Counter()
    used_vertices: set[int] = set()
    face_vertices_by_component: list[int] = []
    degenerate_faces = 0
    offset = 0
    for count in counts:
        face = [int(index) for index in indices[offset : offset + count]]
        offset += count
        if len(face) < 3 or len(set(face)) < 3:
            degenerate_faces += 1
        if not face:
            continue
        face_vertices_by_component.append(face[0])
        used_vertices.update(face)
        for a, b in zip(face, face[1:] + face[:1]):
            if a == b:
                continue
            union(a, b)
            edge = (min(a, b), max(a, b))
            edges[edge] += 1
            edge_orientations[edge] += 1 if a < b else -1

    component_vertices: Counter[int] = Counter(find(index) for index in used_vertices)
    component_faces: Counter[int] = Counter(find(index) for index in face_vertices_by_component)
    largest_components = [
        {
            "vertices": vertex_count,
            "faces": component_faces[root],
        }
        for root, vertex_count in component_vertices.most_common(10)
    ]
    boundary_edges = sum(count == 1 for count in edges.values())
    nonmanifold_edges = sum(count > 2 for count in edges.values())
    inconsistent_edges = sum(
        count == 2 and abs(edge_orientations[edge]) == 2
        for edge, count in edges.items()
    )
    return {
        "edges": len(edges),
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "inconsistent_winding_edges": inconsistent_edges,
        "degenerate_faces": degenerate_faces,
        "connected_components": len(component_vertices),
        "isolated_vertices": len(points) - len(used_vertices),
        "largest_components": largest_components,
        "watertight_manifold": boundary_edges == 0 and nonmanifold_edges == 0,
    }


def _material_source(material: UsdShade.Material, method: str) -> UsdShade.Shader:
    result = getattr(material, method)()
    return result[0] if isinstance(result, tuple) else result


def inspect(path: Path) -> dict[str, Any]:
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {path}")

    prim_counts: Counter[str] = Counter()
    meshes: list[dict[str, Any]] = []
    subsets: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    cameras: list[dict[str, Any]] = []
    lights: list[dict[str, Any]] = []
    skeletons: list[str] = []
    animations: list[str] = []
    blend_shapes: list[str] = []
    instances: list[str] = []
    assets: dict[str, dict[str, Any]] = {}

    for prim in stage.Traverse():
        type_name = prim.GetTypeName() or "<untyped>"
        prim_counts[type_name] += 1
        prim_path = str(prim.GetPath())

        if prim.IsInstance() or prim.IsInstanceProxy():
            instances.append(prim_path)

        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = _value(mesh.GetPointsAttr(), [])
            counts = _value(mesh.GetFaceVertexCountsAttr(), [])
            indices = _value(mesh.GetFaceVertexIndicesAttr(), [])
            normals = _value(mesh.GetNormalsAttr(), [])
            primvars = []
            for primvar in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
                primvars.append(
                    {
                        "name": primvar.GetPrimvarName(),
                        "type": str(primvar.GetTypeName()),
                        "interpolation": str(primvar.GetInterpolation()),
                        "indexed": primvar.IsIndexed(),
                        "values": len(_value(primvar.GetAttr(), [])),
                    }
                )
            meshes.append(
                {
                    "path": prim_path,
                    "points": len(points),
                    "faces": len(counts),
                    "face_vertices": len(indices),
                    "empty": not points or not counts or not indices,
                    "subdivision": str(_value(mesh.GetSubdivisionSchemeAttr(), "catmullClark")),
                    "orientation": str(_value(mesh.GetOrientationAttr(), "rightHanded")),
                    "normals": len(normals),
                    "normals_interpolation": str(mesh.GetNormalsInterpolation()),
                    "primvars": primvars,
                    "material": _bound_material(prim),
                    "skel_api": prim.HasAPI(UsdSkel.BindingAPI),
                    "blend_shape_targets": len(
                        UsdSkel.BindingAPI(prim).GetBlendShapeTargetsRel().GetTargets()
                    ),
                    "world_position": _world_position(prim),
                    "world_bounds": _world_bounds(prim),
                    "topology": _topology(points, counts, indices),
                }
            )

        if prim.IsA(UsdGeom.Subset):
            subset = UsdGeom.Subset(prim)
            subsets.append(
                {
                    "path": prim_path,
                    "element_type": str(_value(subset.GetElementTypeAttr(), "face")),
                    "indices": len(_value(subset.GetIndicesAttr(), [])),
                    "material": _bound_material(prim),
                }
            )

        if prim.IsA(UsdShade.Material):
            material = UsdShade.Material(prim)
            shader = _material_source(material, "ComputeSurfaceSource")
            shader_id = _value(shader.GetIdAttr()) if shader else None
            inputs = []
            if shader:
                for shader_input in shader.GetInputs():
                    sources, _ = UsdShade.ConnectableAPI.GetConnectedSources(shader_input)
                    inputs.append(
                        {
                            "name": str(shader_input.GetBaseName()),
                            "value": str(shader_input.Get()),
                            "sources": [
                                f"{source.source.GetPrim().GetPath()}.{source.sourceName}"
                                for source in sources
                            ],
                        }
                    )
            materials.append(
                {
                    "path": prim_path,
                    "surface_shader": str(shader.GetPath()) if shader else None,
                    "shader_id": str(shader_id) if shader_id else None,
                    "inputs": inputs,
                    "has_displacement": bool(
                        _material_source(material, "ComputeDisplacementSource")
                    ),
                    "has_volume": bool(_material_source(material, "ComputeVolumeSource")),
                }
            )

        if prim.IsA(UsdGeom.Camera):
            camera = UsdGeom.Camera(prim)
            cameras.append(
                {
                    "path": prim_path,
                    "projection": str(_value(camera.GetProjectionAttr())),
                    "focal_length": _value(camera.GetFocalLengthAttr()),
                    "horizontal_aperture": _value(camera.GetHorizontalApertureAttr()),
                    "clipping_range": list(_value(camera.GetClippingRangeAttr(), [])),
                    "world_position": _world_position(prim),
                }
            )

        if prim.HasAPI(UsdLux.LightAPI):
            light = UsdLux.LightAPI(prim)
            entry = {
                "path": prim_path,
                "type": type_name,
                "intensity": _value(light.GetIntensityAttr(), 1.0),
                "exposure": _value(light.GetExposureAttr(), 0.0),
                "color": list(_value(light.GetColorAttr(), [])),
                "world_position": _world_position(prim),
            }
            if prim.IsA(UsdLux.SphereLight):
                sphere = UsdLux.SphereLight(prim)
                entry["radius"] = _value(sphere.GetRadiusAttr(), 0.5)
                entry["treat_as_point"] = _value(sphere.GetTreatAsPointAttr(), False)
            if prim.IsA(UsdLux.DomeLight):
                entry["texture"] = _asset_path(
                    _value(UsdLux.DomeLight(prim).GetTextureFileAttr())
                )
            lights.append(entry)

        if prim.IsA(UsdSkel.Skeleton):
            skeletons.append(prim_path)
        if prim.IsA(UsdSkel.Animation):
            animations.append(prim_path)
        if prim.IsA(UsdSkel.BlendShape):
            blend_shapes.append(prim_path)

        for attribute in prim.GetAttributes():
            asset = _asset_path(_value(attribute))
            if not asset:
                continue
            authored = str(_value(attribute).path)
            resolved = Path(asset)
            if not resolved.is_absolute():
                resolved = (path.parent / resolved).resolve()
            assets[f"{prim_path}.{attribute.GetName()}"] = {
                "authored": authored,
                "resolved": str(resolved),
                "exists": resolved.is_file(),
            }

    return {
        "stage": {
            "path": str(path.resolve()),
            "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "start_time": stage.GetStartTimeCode(),
            "end_time": stage.GetEndTimeCode(),
            "time_codes_per_second": stage.GetTimeCodesPerSecond(),
        },
        "prim_counts": dict(prim_counts.most_common()),
        "meshes": meshes,
        "geom_subsets": subsets,
        "materials": materials,
        "cameras": cameras,
        "lights": lights,
        "skeletons": skeletons,
        "animations": animations,
        "blend_shapes": blend_shapes,
        "instances": instances,
        "assets": assets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a USD scene for renderer compatibility.")
    parser.add_argument("scene", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = inspect(args.scene)
    output = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()

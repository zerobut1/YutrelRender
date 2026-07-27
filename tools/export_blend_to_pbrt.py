from __future__ import annotations

import argparse
import math
import re
import sys
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector


@dataclass(frozen=True)
class ExportedMesh:
    object_name: str
    relative_path: str


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"Non-finite scene value: {value}")
    return f"{value:.9g}"


def _vector(value: Vector) -> str:
    return " ".join(_fmt(component) for component in value)


def _safe_name(name: str, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_name).strip("_")
    return result or fallback


def _render_meshes() -> list[bpy.types.Object]:
    return sorted(
        (
            obj
            for obj in bpy.context.scene.objects
            if obj.type == "MESH" and not obj.hide_render
        ),
        key=lambda obj: obj.name,
    )


@contextmanager
def _export_object(source: bpy.types.Object) -> Iterator[bpy.types.Object]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = source.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(
        evaluated,
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )
    obj = bpy.data.objects.new(f"{source.name}_PBRT_EXPORT", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = source.matrix_world.copy()

    sharp_edges = [edge.index for edge in mesh.edges if edge.use_edge_sharp]
    if sharp_edges:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.edges.ensure_lookup_table()
        bmesh.ops.split_edges(bm, edges=[bm.edges[index] for index in sharp_edges])
        bm.normal_update()
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
    try:
        yield obj
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def _export_meshes(output_dir: Path) -> list[ExportedMesh]:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    view_layer = bpy.context.view_layer
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    results: list[ExportedMesh] = []
    for index, source in enumerate(_render_meshes()):
        stem = _safe_name(source.name, "mesh")
        filename = f"{index:03d}_{stem}.ply"
        path = model_dir / filename
        with _export_object(source) as obj:
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            view_layer.objects.active = obj
            status = bpy.ops.wm.ply_export(
                filepath=str(path),
                check_existing=False,
                forward_axis="Y",
                up_axis="Z",
                global_scale=1.0,
                apply_modifiers=False,
                export_selected_objects=True,
                export_uv=False,
                export_normals=True,
                export_colors="NONE",
                export_attributes=False,
                export_triangulated_mesh=True,
                ascii_format=False,
            )
        if "FINISHED" not in status:
            raise RuntimeError(f"PLY export failed for Blender object '{source.name}'.")
        results.append(
            ExportedMesh(
                object_name=source.name,
                relative_path=f"models/{filename}",
            )
        )
    return results


def _camera_block(scene: bpy.types.Scene, width: int, height: int) -> list[str]:
    camera_obj = scene.camera
    if camera_obj is None or camera_obj.type != "CAMERA":
        raise RuntimeError("The Blender scene must have an active camera.")
    camera = camera_obj.data
    if camera.type != "PERSP":
        raise RuntimeError("Only perspective Blender cameras can be exported.")
    if abs(camera.shift_x) > 1e-6 or abs(camera.shift_y) > 1e-6:
        raise RuntimeError("Yutrel PBRT cameras do not support Blender lens shift.")
    if scene.render.pixel_aspect_x != scene.render.pixel_aspect_y:
        raise RuntimeError("Yutrel PBRT cameras require square render pixels.")

    rotation = camera_obj.matrix_world.to_quaternion()
    eye = camera_obj.matrix_world.translation
    direction = rotation @ Vector((0.0, 0.0, -1.0))
    up = rotation @ Vector((0.0, 1.0, 0.0))
    target = eye + direction

    frame = camera.view_frame(scene=scene)
    tangent = max(abs(corner.y / corner.z) for corner in frame)
    vertical_fov = math.degrees(2.0 * math.atan(tangent))
    return [
        "# Compensate Yutrel's PBRT +Z to -Z camera conversion on the X axis.",
        "Scale -1 1 1",
        f"LookAt {_vector(eye)} {_vector(target)} {_vector(up)}",
        'Camera "perspective"',
        f'    "float fov" [ {_fmt(vertical_fov)} ]',
        "",
        'Film "rgb"',
        '    "string filename" [ "scene.exr" ]',
        f'    "integer xresolution" [ {width} ]',
        f'    "integer yresolution" [ {height} ]',
    ]


def _light_power(light: bpy.types.Light) -> float:
    return light.energy * math.exp2(getattr(light, "exposure", 0.0))


def _point_light(light_obj: bpy.types.Object) -> list[str]:
    light = light_obj.data
    power = _light_power(light)
    intensity = Vector(light.color) * (power / (4.0 * math.pi))
    position = light_obj.matrix_world.translation
    return [
        f"# Delta point light (Blender shadow_soft_size ignored): {light_obj.name}",
        "AttributeBegin",
        f"    Translate {_vector(position)}",
        '    LightSource "point"',
        f'        "rgb I" [ {_vector(intensity)} ]',
        "AttributeEnd",
    ]


def _sun_light(light_obj: bpy.types.Object) -> list[str]:
    light = light_obj.data
    rotation = light_obj.matrix_world.to_quaternion()
    ray_direction = rotation @ Vector((0.0, 0.0, -1.0))
    direction_to_light = -ray_direction.normalized()
    color = Vector(light.color)
    return [
        f"# Sun light: {light_obj.name}",
        'LightSource "distant"',
        f'    "rgb L" [ {_vector(color)} ]',
        f'    "float scale" [ {_fmt(_light_power(light))} ]',
        f'    "point3 from" [ {_vector(direction_to_light)} ]',
        '    "point3 to" [ 0 0 0 ]',
    ]


def _area_light(light_obj: bpy.types.Object) -> list[str]:
    light = light_obj.data
    if light.shape not in {"SQUARE", "RECTANGLE"}:
        raise RuntimeError(
            f"Area light '{light_obj.name}' uses unsupported shape '{light.shape}'. "
            "Use Square or Rectangle."
        )
    size_x = float(light.size)
    size_y = float(light.size if light.shape == "SQUARE" else light.size_y)
    corners = (
        Vector((-0.5 * size_x, -0.5 * size_y, 0.0)),
        Vector((0.5 * size_x, -0.5 * size_y, 0.0)),
        Vector((0.5 * size_x, 0.5 * size_y, 0.0)),
        Vector((-0.5 * size_x, 0.5 * size_y, 0.0)),
    )
    world_corners = [light_obj.matrix_world @ corner for corner in corners]
    edge_x = world_corners[1] - world_corners[0]
    edge_y = world_corners[3] - world_corners[0]
    area = edge_x.cross(edge_y).length
    if area <= 0.0:
        raise RuntimeError(f"Area light '{light_obj.name}' has zero area.")
    radiance = _light_power(light) / (math.pi * area)
    color = Vector(light.color) * radiance
    points = "  ".join(_vector(point) for point in world_corners)
    return [
        f"# Area light: {light_obj.name}",
        "AttributeBegin",
        '    AreaLightSource "diffuse"',
        f'        "rgb L" [ {_vector(color)} ]',
        '    NamedMaterial "LightBlack"',
        '    Shape "trianglemesh"',
        f'        "point3 P" [ {points} ]',
        '        "integer indices" [ 0 2 1  0 3 2 ]',
        "AttributeEnd",
    ]


def _light_blocks(scene: bpy.types.Scene) -> list[str]:
    lines: list[str] = []
    lights = sorted(
        (obj for obj in scene.objects if obj.type == "LIGHT" and not obj.hide_render),
        key=lambda obj: obj.name,
    )
    for light_obj in lights:
        light_type = light_obj.data.type
        if light_type == "POINT":
            block = _point_light(light_obj)
        elif light_type == "SUN":
            block = _sun_light(light_obj)
        elif light_type == "AREA":
            block = _area_light(light_obj)
        else:
            raise RuntimeError(
                f"Unsupported Blender light '{light_obj.name}' of type '{light_type}'."
            )
        lines.extend(block)
        lines.append("")
    return lines


def _world_block(scene: bpy.types.Scene) -> list[str]:
    world = scene.world
    if world is None or not world.use_nodes or world.node_tree is None:
        return []
    output = next(
        (node for node in world.node_tree.nodes if node.type == "OUTPUT_WORLD"),
        None,
    )
    if output is None:
        return []
    surface = output.inputs.get("Surface")
    if surface is None or not surface.is_linked:
        return []
    source = surface.links[0].from_node
    if source.type != "BACKGROUND":
        raise RuntimeError("Only a constant Blender World Background is supported.")
    color_input = source.inputs.get("Color")
    strength_input = source.inputs.get("Strength")
    if color_input is None or strength_input is None:
        return []
    if color_input.is_linked or strength_input.is_linked:
        raise RuntimeError("Linked Blender World Background inputs are not supported.")
    strength = float(strength_input.default_value)
    if strength <= 0.0:
        return []
    color = Vector(color_input.default_value[:3])
    return [
        "# Constant Blender world background",
        'LightSource "infinite"',
        f'    "rgb L" [ {_vector(color)} ]',
        f'    "float scale" [ {_fmt(strength)} ]',
        "",
    ]


def _write_scene(
    path: Path,
    meshes: list[ExportedMesh],
    *,
    samples: int,
    max_depth: int,
    resolution_scale: float,
) -> None:
    scene = bpy.context.scene
    percentage = scene.render.resolution_percentage / 100.0
    width = max(
        1,
        round(scene.render.resolution_x * percentage * resolution_scale),
    )
    height = max(
        1,
        round(scene.render.resolution_y * percentage * resolution_scale),
    )
    lines = [
        "# Generated by tools/export_blend_to_pbrt.py",
        f"# Blender source: {bpy.data.filepath}",
        "",
        'Integrator "path"',
        f'    "integer maxdepth" [ {max_depth} ]',
        'Sampler "zsobol"',
        f'    "integer pixelsamples" [ {samples} ]',
        'PixelFilter "gaussian"',
        '    "float xradius" [ 1.5 ]',
        '    "float yradius" [ 1.5 ]',
        '    "float sigma" [ 0.5 ]',
        "",
    ]
    lines.extend(_camera_block(scene, width, height))
    lines.extend(
        [
            "",
            "WorldBegin",
            "",
            'MakeNamedMaterial "WhiteDiffuse"',
            '    "string type" [ "diffuse" ]',
            '    "rgb reflectance" [ 1 1 1 ]',
            'MakeNamedMaterial "LightBlack"',
            '    "string type" [ "diffuse" ]',
            '    "rgb reflectance" [ 0 0 0 ]',
            "",
        ]
    )
    lines.extend(_world_block(scene))
    lines.extend(_light_blocks(scene))
    for mesh in meshes:
        lines.extend(
            [
                f"# Blender mesh: {mesh.object_name}",
                "AttributeBegin",
                '    NamedMaterial "WhiteDiffuse"',
                '    Shape "plymesh"',
                f'        "string filename" [ "{mesh.relative_path}" ]',
                "AttributeEnd",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the active Blender scene to Yutrel's PBRT subset."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spp", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--resolution-scale", type=float, default=1.0)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    if args.spp <= 0:
        parser.error("--spp must be positive")
    if args.max_depth <= 0:
        parser.error("--max-depth must be positive")
    if not 0.0 < args.resolution_scale <= 1.0:
        parser.error("--resolution-scale must be in (0, 1]")
    return args


def main() -> None:
    args = _arguments()
    output = args.output.resolve()
    if output.suffix.lower() != ".pbrt":
        raise ValueError("--output must name a .pbrt file.")
    output.parent.mkdir(parents=True, exist_ok=True)
    meshes = _export_meshes(output.parent)
    if not meshes:
        raise RuntimeError("The Blender scene has no renderable mesh objects.")
    _write_scene(
        output,
        meshes,
        samples=args.spp,
        max_depth=args.max_depth,
        resolution_scale=args.resolution_scale,
    )
    print(
        f"Exported {len(meshes)} mesh objects and "
        f"{len([obj for obj in bpy.context.scene.objects if obj.type == 'LIGHT' and not obj.hide_render])} lights "
        f"to {output}"
    )


if __name__ == "__main__":
    main()

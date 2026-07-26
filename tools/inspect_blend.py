from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import bpy


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "to_list"):
        return value.to_list()
    try:
        return list(value)
    except TypeError:
        return str(value)


def _custom_properties(value: Any) -> dict[str, Any]:
    return {
        key: _json_value(value[key])
        for key in value.keys()
        if key != "_RNA_UI"
    }


def _node_input(socket: Any) -> dict[str, Any]:
    return {
        "name": socket.name,
        "identifier": socket.identifier,
        "linked": socket.is_linked,
        "value": _json_value(getattr(socket, "default_value", None)),
    }


def _node(node: Any) -> dict[str, Any]:
    result = {
        "name": node.name,
        "label": node.label,
        "type": node.type,
        "bl_idname": node.bl_idname,
        "muted": node.mute,
        "inputs": [_node_input(socket) for socket in node.inputs],
    }
    if node.type == "TEX_IMAGE" and node.image:
        image = node.image
        path = Path(bpy.path.abspath(image.filepath)) if image.filepath else None
        result["image"] = {
            "name": image.name,
            "source": image.source,
            "filepath": image.filepath,
            "resolved": str(path) if path else None,
            "exists": path.is_file() if path else False,
            "packed": image.packed_file is not None,
            "size": list(image.size),
            "channels": image.channels,
            "depth": image.depth,
            "colorspace": image.colorspace_settings.name,
            "alpha_mode": image.alpha_mode,
            "interpolation": node.interpolation,
            "extension": node.extension,
        }
    if node.type == "GROUP" and node.node_tree:
        result["node_group"] = node.node_tree.name
    return result


def _material(material: Any) -> dict[str, Any]:
    result = {
        "name": material.name,
        "use_nodes": material.use_nodes,
        "surface_render_method": getattr(material, "surface_render_method", None),
        "diffuse_color": list(material.diffuse_color),
        "custom_properties": _custom_properties(material),
        "nodes": [],
        "links": [],
    }
    if not material.use_nodes or not material.node_tree:
        return result
    result["nodes"] = [_node(node) for node in material.node_tree.nodes]
    result["links"] = [
        {
            "from_node": link.from_node.name,
            "from_socket": link.from_socket.name,
            "to_node": link.to_node.name,
            "to_socket": link.to_socket.name,
        }
        for link in material.node_tree.links
    ]
    return result


def _material_compatibility(material: Any) -> dict[str, Any]:
    result = {
        "name": material.name,
        "surface_source": None,
        "surface_source_type": None,
        "principled_is_surface": False,
        "base_texture": None,
        "toon_texture": None,
        "sphere_texture": None,
        "mmd_inputs": {},
    }
    if not material.node_tree:
        return result
    for link in material.node_tree.links:
        if link.to_node.type == "OUTPUT_MATERIAL" and link.to_socket.name == "Surface":
            result["surface_source"] = link.from_node.name
            result["surface_source_type"] = link.from_node.bl_idname
            result["principled_is_surface"] = link.from_node.type == "BSDF_PRINCIPLED"
    texture_nodes = {
        "mmd_base_tex": "base_texture",
        "mmd_toon_tex": "toon_texture",
        "mmd_sphere_tex": "sphere_texture",
    }
    for node_name, output_name in texture_nodes.items():
        node = material.node_tree.nodes.get(node_name)
        if node and node.type == "TEX_IMAGE" and node.image:
            path = Path(bpy.path.abspath(node.image.filepath))
            result[output_name] = {
                "image": node.image.name,
                "path": str(path),
                "exists": path.is_file(),
                "packed": node.image.packed_file is not None,
                "channels": node.image.channels,
            }
    shader = material.node_tree.nodes.get("mmd_shader")
    if shader:
        for name in (
            "Ambient Color",
            "Diffuse Color",
            "Specular Color",
            "Reflect",
            "Double Sided",
            "Alpha",
        ):
            socket = shader.inputs.get(name)
            if socket:
                result["mmd_inputs"][name] = _json_value(socket.default_value)
    return result


def _mesh_topology(obj: Any) -> dict[str, Any]:
    mesh = obj.data
    parent = list(range(len(mesh.vertices)))
    ranks = [0] * len(mesh.vertices)

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

    edge_faces: Counter[tuple[int, int]] = Counter()
    edge_orientations: Counter[tuple[int, int]] = Counter()
    used_vertices: set[int] = set()
    component_face_vertices: list[int] = []
    degenerate_faces = 0
    for polygon in mesh.polygons:
        face = list(polygon.vertices)
        if len(face) < 3 or len(set(face)) < 3 or polygon.area <= 1e-12:
            degenerate_faces += 1
        if not face:
            continue
        component_face_vertices.append(face[0])
        used_vertices.update(face)
        for a, b in zip(face, face[1:] + face[:1]):
            if a == b:
                continue
            union(a, b)
            edge = (min(a, b), max(a, b))
            edge_faces[edge] += 1
            edge_orientations[edge] += 1 if a < b else -1

    component_vertices: Counter[int] = Counter(find(index) for index in used_vertices)
    component_faces: Counter[int] = Counter(
        find(index) for index in component_face_vertices
    )
    component_boundary_edges: Counter[int] = Counter()
    component_nonmanifold_edges: Counter[int] = Counter()
    for edge, count in edge_faces.items():
        root = find(edge[0])
        if count == 1:
            component_boundary_edges[root] += 1
        elif count > 2:
            component_nonmanifold_edges[root] += 1
    boundary_edges = sum(count == 1 for count in edge_faces.values())
    nonmanifold_edges = sum(count > 2 for count in edge_faces.values())
    inconsistent_edges = sum(
        count == 2 and abs(edge_orientations[edge]) == 2
        for edge, count in edge_faces.items()
    )

    world_points = [obj.matrix_world @ vertex.co for vertex in mesh.vertices]
    minimum = [min(point[axis] for point in world_points) for axis in range(3)]
    maximum = [max(point[axis] for point in world_points) for axis in range(3)]
    origin = obj.matrix_world.translation
    return {
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "inconsistent_winding_edges": inconsistent_edges,
        "degenerate_faces": degenerate_faces,
        "connected_components": len(component_vertices),
        "watertight_components": sum(
            component_boundary_edges[root] == 0
            and component_nonmanifold_edges[root] == 0
            for root in component_vertices
        ),
        "open_components": sum(
            component_boundary_edges[root] != 0
            or component_nonmanifold_edges[root] != 0
            for root in component_vertices
        ),
        "isolated_vertices": len(mesh.vertices) - len(used_vertices),
        "largest_components": [
            {
                "vertices": vertex_count,
                "faces": component_faces[root],
                "boundary_edges": component_boundary_edges[root],
                "nonmanifold_edges": component_nonmanifold_edges[root],
            }
            for root, vertex_count in component_vertices.most_common(10)
        ],
        "watertight_manifold": boundary_edges == 0 and nonmanifold_edges == 0,
        "world_bounds": {
            "min": minimum,
            "max": maximum,
            "size": [maximum[axis] - minimum[axis] for axis in range(3)],
        },
        "world_origin": list(origin),
        "origin_height_above_min_z": float(origin.z - minimum[2]),
    }


def _mesh_object(obj: Any) -> dict[str, Any]:
    mesh = obj.data
    material_faces = Counter(poly.material_index for poly in mesh.polygons)
    material_slots = []
    for index, slot in enumerate(obj.material_slots):
        material_slots.append(
            {
                "index": index,
                "name": slot.material.name if slot.material else None,
                "link": slot.link,
                "faces": material_faces[index],
            }
        )
    attributes = [
        {
            "name": attribute.name,
            "domain": attribute.domain,
            "data_type": attribute.data_type,
            "values": len(attribute.data),
        }
        for attribute in mesh.attributes
        if not attribute.is_internal
    ]
    shape_keys = []
    if mesh.shape_keys:
        shape_keys = [key.name for key in mesh.shape_keys.key_blocks]
    return {
        "data_name": mesh.name,
        "vertices": len(mesh.vertices),
        "edges": len(mesh.edges),
        "polygons": len(mesh.polygons),
        "loops": len(mesh.loops),
        "triangles": sum(max(len(poly.vertices) - 2, 0) for poly in mesh.polygons),
        "material_slots": material_slots,
        "uv_layers": [layer.name for layer in mesh.uv_layers],
        "active_uv": mesh.uv_layers.active.name if mesh.uv_layers.active else None,
        "color_attributes": [attribute.name for attribute in mesh.color_attributes],
        "attributes": attributes,
        "vertex_groups": [group.name for group in obj.vertex_groups],
        "shape_keys": shape_keys,
        "use_auto_smooth": getattr(mesh, "use_auto_smooth", None),
        "topology": _mesh_topology(obj),
    }


def _object(obj: Any) -> dict[str, Any]:
    result = {
        "name": obj.name,
        "type": obj.type,
        "data_name": obj.data.name if obj.data else None,
        "parent": obj.parent.name if obj.parent else None,
        "collections": [collection.name for collection in obj.users_collection],
        "hide_render": obj.hide_render,
        "hide_viewport": obj.hide_viewport,
        "location": list(obj.location),
        "rotation_mode": obj.rotation_mode,
        "rotation_euler": list(obj.rotation_euler),
        "scale": list(obj.scale),
        "dimensions": list(obj.dimensions),
        "modifiers": [
            {
                "name": modifier.name,
                "type": modifier.type,
                "show_render": modifier.show_render,
                "object": getattr(getattr(modifier, "object", None), "name", None),
            }
            for modifier in obj.modifiers
        ],
        "custom_properties": _custom_properties(obj),
    }
    if obj.type == "MESH":
        result["mesh"] = _mesh_object(obj)
    elif obj.type == "LIGHT":
        light = obj.data
        result["light"] = {
            "type": light.type,
            "energy": light.energy,
            "color": list(light.color),
            "shadow_soft_size": light.shadow_soft_size,
            "use_nodes": light.use_nodes,
        }
    elif obj.type == "CAMERA":
        camera = obj.data
        result["camera"] = {
            "type": camera.type,
            "lens": camera.lens,
            "sensor_fit": camera.sensor_fit,
            "sensor_width": camera.sensor_width,
            "sensor_height": camera.sensor_height,
            "shift_x": camera.shift_x,
            "shift_y": camera.shift_y,
            "clip_start": camera.clip_start,
            "clip_end": camera.clip_end,
            "dof_enabled": camera.dof.use_dof,
            "focus_distance": camera.dof.focus_distance,
            "aperture_fstop": camera.dof.aperture_fstop,
        }
    elif obj.type == "ARMATURE":
        result["armature"] = {
            "bones": len(obj.data.bones),
            "pose_bones": len(obj.pose.bones),
        }
    return result


def _world(world: Any) -> dict[str, Any] | None:
    if not world:
        return None
    result = {
        "name": world.name,
        "color": list(world.color),
        "use_nodes": world.use_nodes,
        "nodes": [],
        "links": [],
    }
    if world.use_nodes and world.node_tree:
        result["nodes"] = [_node(node) for node in world.node_tree.nodes]
        result["links"] = [
            {
                "from_node": link.from_node.name,
                "from_socket": link.from_socket.name,
                "to_node": link.to_node.name,
                "to_socket": link.to_socket.name,
            }
            for link in world.node_tree.links
        ]
    return result


def inspect() -> dict[str, Any]:
    scene = bpy.context.scene
    return {
        "blender": {
            "version": bpy.app.version_string,
            "filepath": bpy.data.filepath,
        },
        "scene": {
            "name": scene.name,
            "frame_current": scene.frame_current,
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "camera": scene.camera.name if scene.camera else None,
            "render_engine": scene.render.engine,
            "resolution_x": scene.render.resolution_x,
            "resolution_y": scene.render.resolution_y,
            "resolution_percentage": scene.render.resolution_percentage,
            "film_transparent": scene.render.film_transparent,
            "view_transform": scene.view_settings.look,
            "world": _world(scene.world),
        },
        "collections": [
            {
                "name": collection.name,
                "objects": [obj.name for obj in collection.objects],
                "hide_render": collection.hide_render,
                "hide_viewport": collection.hide_viewport,
            }
            for collection in bpy.data.collections
        ],
        "object_counts": dict(Counter(obj.type for obj in bpy.data.objects)),
        "objects": [_object(obj) for obj in bpy.data.objects],
        "materials": [_material(material) for material in bpy.data.materials],
        "material_compatibility": [
            _material_compatibility(material) for material in bpy.data.materials
        ],
        "images": [
            {
                "name": image.name,
                "source": image.source,
                "filepath": image.filepath,
                "resolved": bpy.path.abspath(image.filepath) if image.filepath else None,
                "packed": image.packed_file is not None,
                "size": list(image.size),
                "colorspace": image.colorspace_settings.name,
            }
            for image in bpy.data.images
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--material-csv", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    report = inspect()
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if args.material_csv:
        with args.material_csv.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ("material", "surface", "base", "toon", "sphere", "alpha", "double_sided")
            )
            for material in report["material_compatibility"]:
                writer.writerow(
                    (
                        material["name"],
                        material["surface_source"],
                        (material["base_texture"] or {}).get("image"),
                        (material["toon_texture"] or {}).get("image"),
                        (material["sphere_texture"] or {}).get("image"),
                        material["mmd_inputs"].get("Alpha"),
                        material["mmd_inputs"].get("Double Sided"),
                    )
                )


if __name__ == "__main__":
    main()

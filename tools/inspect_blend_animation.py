from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import bpy


def _action_summary(action: Any) -> dict[str, Any]:
    curves = list(getattr(action, "fcurves", ()))
    return {
        "name": action.name,
        "frame_range": [float(value) for value in action.frame_range],
        "fcurves": len(curves),
        "keyframes": sum(len(curve.keyframe_points) for curve in curves),
        "slots": len(getattr(action, "slots", ())),
        "layers": len(getattr(action, "layers", ())),
    }


def inspect() -> dict[str, Any]:
    scene = bpy.context.scene
    modifier_counts: Counter[str] = Counter()
    constraints = 0
    pose_constraints = 0
    nla_tracks = 0
    nla_strips = 0
    mesh_vertices = 0
    mesh_polygons = 0
    shape_keys = 0
    armature_bones = 0
    for obj in bpy.data.objects:
        modifier_counts.update(modifier.type for modifier in obj.modifiers)
        constraints += len(obj.constraints)
        animation = obj.animation_data
        if animation:
            nla_tracks += len(animation.nla_tracks)
            nla_strips += sum(len(track.strips) for track in animation.nla_tracks)
        if obj.type == "MESH":
            mesh_vertices += len(obj.data.vertices)
            mesh_polygons += len(obj.data.polygons)
            if obj.data.shape_keys:
                shape_keys += len(obj.data.shape_keys.key_blocks)
        elif obj.type == "ARMATURE":
            armature_bones += len(obj.data.bones)
            pose_constraints += sum(len(bone.constraints) for bone in obj.pose.bones)

    rigidbody_objects = sum(obj.rigid_body is not None for obj in bpy.data.objects)
    rigidbody_constraints = sum(
        obj.rigid_body_constraint is not None for obj in bpy.data.objects
    )
    empty_objects = [obj for obj in bpy.data.objects if obj.type == "EMPTY"]
    rigidbody_samples = [obj for obj in bpy.data.objects if obj.rigid_body is not None][:30]
    return {
        "blender": bpy.app.version_string,
        "filepath": bpy.data.filepath,
        "scene": {
            "name": scene.name,
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "frame_current": scene.frame_current,
            "fps": scene.render.fps,
            "fps_base": scene.render.fps_base,
            "sync_mode": scene.sync_mode,
            "render_engine": scene.render.engine,
            "use_simplify": scene.render.use_simplify,
            "simplify_subdivision": scene.render.simplify_subdivision,
            "simplify_child_particles": scene.render.simplify_child_particles,
        },
        "objects": dict(Counter(obj.type for obj in bpy.data.objects)),
        "mesh_vertices": mesh_vertices,
        "mesh_polygons": mesh_polygons,
        "shape_keys": shape_keys,
        "armature_bones": armature_bones,
        "modifiers": dict(modifier_counts),
        "constraints": constraints,
        "pose_constraints": pose_constraints,
        "nla_tracks": nla_tracks,
        "nla_strips": nla_strips,
        "rigidbody_objects": rigidbody_objects,
        "rigidbody_constraints": rigidbody_constraints,
        "has_rigidbody_world": scene.rigidbody_world is not None,
        "rigidbody_world": (
            {
                "enabled": scene.rigidbody_world.enabled,
                "substeps_per_frame": scene.rigidbody_world.substeps_per_frame,
                "solver_iterations": scene.rigidbody_world.solver_iterations,
                "cache_baked": scene.rigidbody_world.point_cache.is_baked,
                "cache_frame_start": scene.rigidbody_world.point_cache.frame_start,
                "cache_frame_end": scene.rigidbody_world.point_cache.frame_end,
            }
            if scene.rigidbody_world
            else None
        ),
        "collections": sorted(
            (
                {
                    "name": collection.name,
                    "direct_objects": len(collection.objects),
                    "all_objects": len(collection.all_objects),
                    "hide_viewport": collection.hide_viewport,
                    "hide_render": collection.hide_render,
                }
                for collection in bpy.data.collections
            ),
            key=lambda value: value["all_objects"],
            reverse=True,
        )[:30],
        "empty_stats": {
            "hidden_viewport": sum(obj.hide_viewport for obj in empty_objects),
            "hidden_render": sum(obj.hide_render for obj in empty_objects),
            "display_types": dict(Counter(obj.empty_display_type for obj in empty_objects)),
            "parent_types": dict(
                Counter(obj.parent.type if obj.parent else "NONE" for obj in empty_objects)
            ),
            "samples": [
                {
                    "name": obj.name,
                    "parent": obj.parent.name if obj.parent else None,
                    "collections": [collection.name for collection in obj.users_collection],
                    "hidden_viewport": obj.hide_viewport,
                }
                for obj in empty_objects[:30]
            ],
        },
        "rigidbody_samples": [
            {
                "name": obj.name,
                "type": obj.type,
                "rigidbody_type": obj.rigid_body.type,
                "collections": [collection.name for collection in obj.users_collection],
                "hidden_viewport": obj.hide_viewport,
            }
            for obj in rigidbody_samples
        ],
        "actions": [_action_summary(action) for action in bpy.data.actions],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.write_text(
        json.dumps(inspect(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

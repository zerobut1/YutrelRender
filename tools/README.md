# Yutrel regression tools

Run a fast Cornell Box regression:

```text
uv run python tools/render_regression.py projects/Yutrel/scene/cornell-box --profile quick
```

Run the four-seed quality profile:

```text
uv run python tools/render_regression.py projects/Yutrel/scene/cornell-box --profile quality
```

Compare two linear RGB EXR files directly:

```text
uv run python tools/compare_exr.py --actual actual.exr --reference reference.exr --nrmse-limit 0.03 --mean-rgb-bias-limit 0.05
```

The tools never generate or update `reference.exr`. Regression outputs and JSON reports are
written below `build/Yutrel/regression/`.

Split a static PMX model into one binary PLY per material and generate a PBRT-v4 scene:

```text
uv run python tools/pmx_to_pbrt.py path/to/model.pmx
```

Outputs are written to `path/to/pbrt/` by default. The conversion preserves positions,
normals, UVs, base textures, material triangle ranges, and texture alpha masks. Bones,
morphs, and animation are intentionally not exported.

Export the active Blender scene to Yutrel's supported PBRT subset:

```text
blender --background path/to/scene.blend --python tools/export_blend_to_pbrt.py -- --output path/to/scene.pbrt
```

The exporter writes one binary PLY per visible mesh, the active perspective camera,
constant world background, and Point, Sun, or rectangular Area lights. Point lights are
converted to emissive spheres. All scene meshes use a white diffuse material.

using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using YutrelRP;

namespace Yutrel.PathTracer
{
    internal sealed class YutrelPathTracer : YutrelRenderer
    {
        private sealed class CameraResources
        {
            internal RTHandle color;
            internal int width;
            internal int height;

            internal bool Ensure(Camera camera, Vector2Int targetSize)
            {
                if (color != null && width == targetSize.x && height == targetSize.y)
                {
                    return false;
                }

                Release();
                width = targetSize.x;
                height = targetSize.y;
                color = RTHandles.Alloc(
                    targetSize.x,
                    targetSize.y,
                    colorFormat: GraphicsFormat.R16G16B16A16_SFloat,
                    enableRandomWrite: true,
                    filterMode: FilterMode.Bilinear,
                    wrapMode: TextureWrapMode.Clamp,
                    name: $"YutrelPathTracer Color ({camera.name})");
                return true;
            }

            internal void Release()
            {
                color?.Release();
                color = null;
                width = 0;
                height = 0;
            }
        }

        private sealed class ViewState
        {
            internal readonly uint nativeViewId;
            internal readonly CameraResources resources = new();
            internal Matrix4x4 previousCameraToWorld;
            internal float previousVerticalFov;
            internal bool previousCameraValid;

            internal ViewState(uint nativeViewId)
            {
                this.nativeViewId = nativeViewId;
            }

            internal void Release()
            {
                resources.Release();
                previousCameraValid = false;
            }
        }

        private readonly struct ViewKey : IEquatable<ViewKey>
        {
            private readonly ulong cameraId;
            private readonly CameraType cameraType;
            private readonly bool outputsToBackbuffer;

            internal ViewKey(Camera camera)
            {
                cameraId = EntityId.ToULong(camera.GetEntityId());
                cameraType = camera.cameraType;
                outputsToBackbuffer = camera.targetTexture == null;
            }

            public bool Equals(ViewKey other)
            {
                return cameraId == other.cameraId &&
                       cameraType == other.cameraType &&
                       outputsToBackbuffer == other.outputsToBackbuffer;
            }

            public override bool Equals(object obj)
            {
                return obj is ViewKey other && Equals(other);
            }

            public override int GetHashCode()
            {
                unchecked
                {
                    var hash = cameraId.GetHashCode();
                    hash = hash * 397 ^ (int)cameraType;
                    hash = hash * 397 ^ (outputsToBackbuffer ? 1 : 0);
                    return hash;
                }
            }
        }

        private sealed class SceneChangeTracker : IDisposable
        {
            // Periodic full scan interval (both Edit and Play mode).
            // This guarantees material/shader parameter tweaks are picked up
            // within this interval even when ObjectChangeEvents does not fire
            // (e.g. Inspector drags on Material assets).
            private const double StructureScanInterval = 0.25;
            private const double EditorChangeDebounce = 0.15;
            private const string DefaultLitShaderName = "YutrelRP/DefaultLit";
            private const string UnlitShaderName = "YutrelRP/Unlit";
            private const string EmissiveColorProperty = "_Emissive";
            private const string EmissiveLuminanceProperty = "_EmissiveLuminanceNits";
            private const string UseEmissiveTextureProperty = "_UseEmissiveTex";
            private const string EmissiveTextureProperty = "_EmissiveTex";
            private const string MainTextureProperty = "_MainTex";
            private const string CullModeProperty = "_CullMode";

            private sealed class TrackedMesh
            {
                internal MeshFilter filter;
                internal MeshRenderer renderer;
                internal ulong sharedMeshId;
                internal int materialSignature;
                internal SubMeshMaterialState[] subMeshStates;
                internal Matrix4x4 localToWorld;
            }

            // Per-submesh snapshot of everything the renderer consumes from a
            // Material. `restHash` covers all properties except the OpenPBR
            // parameters; `paramsHash` covers the 9 OpenPBR parameters. When a
            // scan finds only `paramsHash` changes, the change can be shipped as
            // a lightweight material-only update (no vertex re-upload, no scene
            // rebuild, no kernel recompile).
            private struct SubMeshMaterialState
            {
                internal ulong materialId;
                internal NativeBridge.SceneMaterialType materialType;
                internal bool doubleSided;
                internal int restHash;
                internal int paramsHash;
            }

            private readonly Dictionary<ulong, TrackedMesh> meshes = new();
            private readonly HashSet<ulong> seenMeshIds = new();
            private readonly List<ulong> removedMeshIds = new();
            private readonly List<NativeBridge.SceneMeshUpdate> pendingUpdates = new();
            private readonly List<NativeBridge.SceneMaterialUpdate> pendingMaterialUpdates = new();
            private readonly List<Material> sharedMaterials = new();

            private Light selectedLight;
            private NativeBridge.DirectionalLightUpdate previousLight;
            private bool previousLightValid;
            private NativeBridge.EnvironmentUpdate previousEnvironment;
            private bool previousEnvironmentValid;
            private bool initialized;
            private bool structureDirty = true;
            private double nextStructureScan;
            private double editorStructureScanNotBefore;
            private bool lastScanAllRebuilt;
            private ulong revision;

            internal SceneChangeTracker()
            {
#if UNITY_EDITOR
                UnityEditor.ObjectChangeEvents.changesPublished += OnObjectChangesPublished;
                UnityEditor.EditorApplication.hierarchyChanged += OnHierarchyChanged;
#endif
            }

            public void Dispose()
            {
#if UNITY_EDITOR
                UnityEditor.ObjectChangeEvents.changesPublished -= OnObjectChangesPublished;
                UnityEditor.EditorApplication.hierarchyChanged -= OnHierarchyChanged;
#endif
                meshes.Clear();
                pendingUpdates.Clear();
                pendingMaterialUpdates.Clear();
            }

            internal bool Synchronize(Camera camera)
            {
                pendingUpdates.Clear();
                pendingMaterialUpdates.Clear();
                var now = Time.realtimeSinceStartupAsDouble;
                // Scan periodically in both Play and Edit mode so that material
                // parameter changes always become visible without relying on
                // editor change events (which are not guaranteed for Material
                // Inspector edits). The native plugin restarts accumulation
                // (1spp) automatically whenever a scene delta is applied.
                var scanDue = now >= nextStructureScan;
                var structureScanReady = !initialized || now >= editorStructureScanNotBefore;
                if (structureScanReady && (!initialized || structureDirty || scanDue))
                {
                    ScanStructure();
                    initialized = true;
                    structureDirty = false;
                    nextStructureScan = now + StructureScanInterval;
                }

                CompareTransforms();
                var light = CurrentLightState();
                var lightChanged = !previousLightValid || !LightNear(previousLight, light);
                var environment = CurrentEnvironmentState(camera);
                var environmentChanged = !previousEnvironmentValid ||
                                         !EnvironmentNear(previousEnvironment, environment);
                if (pendingUpdates.Count == 0 && pendingMaterialUpdates.Count == 0 &&
                    !lightChanged && !environmentChanged)
                {
                    return true;
                }

                revision++;
                if (!NativeBridge.SubmitSceneDelta(
                        pendingUpdates,
                        pendingMaterialUpdates,
                        revision,
                        lightChanged ? light : null,
                        environmentChanged ? environment : null))
                {
                    meshes.Clear();
                    selectedLight = null;
                    previousLightValid = false;
                    previousEnvironmentValid = false;
                    initialized = false;
                    structureDirty = true;
                    return false;
                }
                previousLight = light;
                previousLightValid = true;
                previousEnvironment = environment;
                previousEnvironmentValid = true;
                return true;
            }

            private void ScanStructure()
            {
                // Texture reads (GetPixels) are cached per material for the
                // duration of one structure scan: Sponza shares a handful of
                // materials across hundreds of meshes, so without this cache
                // every submesh re-decodes every map (~1500 GetPixels calls
                // instead of ~96). Cleared each scan so modified textures are
                // re-read on the next full rebuild.
                TextureReadCache.Clear();
                seenMeshIds.Clear();
                var meshCount = 0;
                var rebuiltCount = 0;
                foreach (var filter in UnityEngine.Object.FindObjectsByType<MeshFilter>(
                             FindObjectsInactive.Exclude))
                {
                    var meshRenderer = filter.GetComponent<MeshRenderer>();
                    if (meshRenderer == null || !meshRenderer.enabled ||
                        !filter.gameObject.activeInHierarchy || filter.sharedMesh == null)
                    {
                        continue;
                    }
                    meshCount++;

                    var objectId = EntityId.ToULong(filter.GetEntityId());
                    var sharedMeshId = EntityId.ToULong(filter.sharedMesh.GetEntityId());
                    var materialSignature = ComputeMaterialSignature(filter.sharedMesh, meshRenderer);
                    if (meshes.TryGetValue(objectId, out var tracked) &&
                        tracked.sharedMeshId == sharedMeshId &&
                        tracked.materialSignature == materialSignature &&
                        filter.sharedMesh.isReadable &&
                        IsValidTransform(filter.transform.localToWorldMatrix))
                    {
                        tracked.filter = filter;
                        tracked.renderer = meshRenderer;
                        seenMeshIds.Add(objectId);
                        continue;
                    }

                    // When only OpenPBR parameter values changed (geometry and
                    // every other material property unchanged), ship lightweight
                    // material-only updates instead of re-uploading all vertices:
                    // the native plugin uploads new values into 1x1 bindless
                    // parameter images without rebuilding the scene or
                    // recompiling the kernel.
                    List<NativeBridge.SceneMaterialUpdate> materialUpdates = null;
                    var materialOnly = tracked != null &&
                                       tracked.sharedMeshId == sharedMeshId &&
                                       TryBuildMaterialUpdates(
                                           filter,
                                           objectId,
                                           tracked,
                                           out materialUpdates);
                    if (materialOnly)
                    {
                        pendingMaterialUpdates.AddRange(materialUpdates);
                        tracked.filter = filter;
                        tracked.renderer = meshRenderer;
                        tracked.materialSignature = materialSignature;
                        tracked.localToWorld = filter.transform.localToWorldMatrix;
                        seenMeshIds.Add(objectId);
                        continue;
                    }

                    if (!TryCreateMeshUpdate(filter, out var update))
                    {
                        continue;
                    }
                    rebuiltCount++;
                    pendingUpdates.Add(update);
                    var localToWorld = filter.transform.localToWorldMatrix;
                    if (tracked == null)
                    {
                        tracked = new TrackedMesh();
                        meshes.Add(objectId, tracked);
                    }
                    tracked.filter = filter;
                    tracked.renderer = meshRenderer;
                    tracked.sharedMeshId = sharedMeshId;
                    tracked.materialSignature = materialSignature;
                    tracked.subMeshStates = ComputeSubMeshMaterialStates(filter);
                    tracked.localToWorld = localToWorld;
                    seenMeshIds.Add(objectId);
                }

                removedMeshIds.Clear();
                foreach (var pair in meshes)
                {
                    if (!seenMeshIds.Contains(pair.Key))
                    {
                        pendingUpdates.Add(NativeBridge.SceneMeshUpdate.Remove(pair.Key));
                        removedMeshIds.Add(pair.Key);
                    }
                }
                foreach (var objectId in removedMeshIds)
                {
                    meshes.Remove(objectId);
                }

                // Diagnostic: when every mesh is rebuilt on every scan, some
                // identity or signature input is unstable (e.g. a non-stable
                // object id), which would re-read and re-upload all textures
                // every 0.25 s. Report once so the cause is visible in logs.
                var allRebuilt = meshCount > 0 && rebuiltCount == meshCount;
                if (allRebuilt && lastScanAllRebuilt)
                {
                    NativeBridge.ReportErrorOnce(
                        $"Yutrel rebuilt all {meshCount} meshes again this scan; " +
                        "object/material identity or signatures are unstable, " +
                        "causing repeated full scene uploads.");
                }
                lastScanAllRebuilt = allRebuilt;

                selectedLight = null;
                var directionalLightCount = 0;
                foreach (var light in UnityEngine.Object.FindObjectsByType<Light>(
                             FindObjectsInactive.Exclude))
                {
                    if (!light.enabled || !light.gameObject.activeInHierarchy ||
                        light.type != LightType.Directional)
                    {
                        continue;
                    }
                    selectedLight = light;
                    directionalLightCount++;
                }
                if (directionalLightCount != 1)
                {
                    selectedLight = null;
                    NativeBridge.ReportInfoOnce(
                        directionalLightCount == 0
                            ? "YutrelPathTracer lighting is disabled until one Directional Light is enabled."
                            : "YutrelPathTracer lighting is disabled while multiple Directional Lights are enabled.");
                }
            }

            private void CompareTransforms()
            {
                removedMeshIds.Clear();
                foreach (var pair in meshes)
                {
                    var tracked = pair.Value;
                    if (tracked.filter == null || tracked.renderer == null ||
                        !tracked.renderer.enabled || !tracked.filter.gameObject.activeInHierarchy)
                    {
                        structureDirty = true;
                        pendingUpdates.Add(NativeBridge.SceneMeshUpdate.Remove(pair.Key));
                        removedMeshIds.Add(pair.Key);
                        continue;
                    }
                    var localToWorld = tracked.filter.transform.localToWorldMatrix;
                    if (!IsValidTransform(localToWorld))
                    {
                        NativeBridge.ReportErrorOnce(
                            $"Yutrel Mesh '{tracked.filter.gameObject.name}' has a singular or non-finite Transform.");
                        structureDirty = true;
                        pendingUpdates.Add(NativeBridge.SceneMeshUpdate.Remove(pair.Key));
                        removedMeshIds.Add(pair.Key);
                        continue;
                    }
                    if (MatrixNear(tracked.localToWorld, localToWorld))
                    {
                        continue;
                    }
                    pendingUpdates.Add(NativeBridge.SceneMeshUpdate.Transform(
                        pair.Key,
                        localToWorld));
                    tracked.localToWorld = localToWorld;
                }
                foreach (var objectId in removedMeshIds)
                {
                    meshes.Remove(objectId);
                }
            }

            private NativeBridge.DirectionalLightUpdate CurrentLightState()
            {
                if (selectedLight == null)
                {
                    return new NativeBridge.DirectionalLightUpdate(
                        Vector3.zero,
                        0.0f,
                        Vector3.forward,
                        false);
                }
                if (!selectedLight.enabled || !selectedLight.gameObject.activeInHierarchy ||
                    selectedLight.type != LightType.Directional)
                {
                    structureDirty = true;
                    return new NativeBridge.DirectionalLightUpdate(
                        Vector3.zero,
                        0.0f,
                        Vector3.forward,
                        false);
                }

                var color = PhotometricColor.NormalizeLinearSrgb(selectedLight.color.linear);
                var illuminanceLux = selectedLight.intensity;
                var direction = -selectedLight.transform.forward;
                if (!IsFinite(color) ||
                    color.x < 0.0f || color.y < 0.0f || color.z < 0.0f ||
                    !IsFinite(illuminanceLux) || illuminanceLux < 0.0f ||
                    !IsFinite(direction) || direction.sqrMagnitude < 1e-12f)
                {
                    NativeBridge.ReportErrorOnce("The Yutrel Directional Light state is invalid.");
                    return new NativeBridge.DirectionalLightUpdate(
                        Vector3.zero,
                        0.0f,
                        Vector3.forward,
                        false);
                }
                direction.Normalize();
                return new NativeBridge.DirectionalLightUpdate(
                    color,
                    illuminanceLux,
                    direction,
                    true);
            }

            private static NativeBridge.EnvironmentUpdate CurrentEnvironmentState(Camera camera)
            {
#if UNITY_EDITOR
                if (!YutrelEnvironmentLight.TryResolve(camera, out var environmentLight))
                {
                    return NativeBridge.EnvironmentUpdate.Disabled;
                }

                var intensity = environmentLight.Intensity;
                if (!IsFinite(intensity) || intensity < 0.0f)
                {
                    NativeBridge.ReportErrorOnce(
                        "YutrelEnvironmentLight has an invalid environment intensity.");
                    return NativeBridge.EnvironmentUpdate.Disabled;
                }
                if (intensity == 0.0f)
                {
                    return NativeBridge.EnvironmentUpdate.Disabled;
                }

                var asset = environmentLight.IblAsset;
                if (asset == null)
                {
                    NativeBridge.ReportErrorOnce(
                        "YutrelEnvironmentLight has no IBL asset for Path Tracing.");
                    return NativeBridge.EnvironmentUpdate.Disabled;
                }

                var sourceTexture = asset.SourceEnvironmentTexture;
                if (sourceTexture == null || sourceTexture.height <= 0 ||
                    sourceTexture.width != sourceTexture.height * 2)
                {
                    NativeBridge.ReportErrorOnce(
                        "YutrelEnvironmentLight Path Tracing source texture must be a 2:1 lat-long HDR.");
                    return NativeBridge.EnvironmentUpdate.Disabled;
                }

                var sourcePath = asset.sourceEnvironmentTexturePath?.Trim();
                if (string.IsNullOrEmpty(sourcePath) ||
                    !sourcePath.Replace('\\', '/').StartsWith(
                        "Assets/", StringComparison.OrdinalIgnoreCase))
                {
                    NativeBridge.ReportErrorOnce(
                        "YutrelEnvironmentLight has no valid project-relative source HDR path.");
                    return NativeBridge.EnvironmentUpdate.Disabled;
                }

                try
                {
                    var projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
                    var assetsRoot = Path.GetFullPath(Application.dataPath);
                    var assetsPrefix = assetsRoot.EndsWith(
                        Path.DirectorySeparatorChar.ToString(), StringComparison.Ordinal)
                        ? assetsRoot
                        : assetsRoot + Path.DirectorySeparatorChar;
                    var hdrPath = Path.GetFullPath(Path.Combine(projectRoot, sourcePath));
                    if (!hdrPath.StartsWith(assetsPrefix, StringComparison.OrdinalIgnoreCase) ||
                        !File.Exists(hdrPath))
                    {
                        NativeBridge.ReportErrorOnce(
                            $"YutrelEnvironmentLight source HDR does not exist under Assets: '{sourcePath}'.");
                        return NativeBridge.EnvironmentUpdate.Disabled;
                    }
                    return new NativeBridge.EnvironmentUpdate(hdrPath, intensity, true);
                }
                catch (Exception exception) when (
                    exception is ArgumentException ||
                    exception is NotSupportedException ||
                    exception is IOException)
                {
                    NativeBridge.ReportErrorOnce(
                        $"YutrelEnvironmentLight source HDR path is invalid: '{sourcePath}'.");
                    return NativeBridge.EnvironmentUpdate.Disabled;
                }
#else
                return NativeBridge.EnvironmentUpdate.Disabled;
#endif
            }

            private bool TryCreateMeshUpdate(
                MeshFilter filter,
                out NativeBridge.SceneMeshUpdate update)
            {
                update = default;
                var mesh = filter.sharedMesh;
                var objectName = filter.gameObject.name;
                if (mesh == null || !mesh.isReadable)
                {
                    NativeBridge.ReportErrorOnce(
                        $"Yutrel Mesh '{objectName}' must exist and have Read/Write enabled.");
                    return false;
                }
                var localToWorld = filter.transform.localToWorldMatrix;
                if (!IsValidTransform(localToWorld))
                {
                    NativeBridge.ReportErrorOnce(
                        $"Yutrel Mesh '{objectName}' has a singular or non-finite Transform.");
                    return false;
                }
                var vertices = mesh.vertices;
                var normals = mesh.normals;
                if (vertices.Length == 0 || normals.Length != vertices.Length)
                {
                    NativeBridge.ReportErrorOnce(
                        $"Yutrel Mesh '{objectName}' must provide one normal per vertex.");
                    return false;
                }
                for (var vertexIndex = 0; vertexIndex < vertices.Length; vertexIndex++)
                {
                    var position = vertices[vertexIndex];
                    var normal = normals[vertexIndex];
                    if (!IsFinite(position) || !IsFinite(normal) || normal.sqrMagnitude < 1e-12f)
                    {
                        NativeBridge.ReportErrorOnce(
                            $"Yutrel Mesh '{objectName}' contains a non-finite vertex or invalid normal.");
                        return false;
                    }
                }

                var uvs = mesh.uv;
                if (uvs.Length != vertices.Length)
                {
                    uvs = null;
                }

                var indices = new List<uint>();
                var subMeshes = new NativeBridge.SceneSubMeshUpdate[mesh.subMeshCount];
                sharedMaterials.Clear();
                filter.GetComponent<MeshRenderer>().GetSharedMaterials(sharedMaterials);
                for (var subMesh = 0; subMesh < mesh.subMeshCount; subMesh++)
                {
                    if (mesh.GetTopology(subMesh) != MeshTopology.Triangles)
                    {
                        NativeBridge.ReportErrorOnce(
                            $"Every submesh of Yutrel Mesh '{objectName}' must use Triangle topology.");
                        return false;
                    }
                    var indexOffset = (uint)indices.Count;
                    foreach (var index in mesh.GetIndices(subMesh, true))
                    {
                        if ((uint)index >= (uint)vertices.Length)
                        {
                            NativeBridge.ReportErrorOnce(
                                $"Yutrel Mesh '{objectName}' contains an out-of-range index.");
                            return false;
                        }
                        indices.Add((uint)index);
                    }
                    var indexCount = (uint)indices.Count - indexOffset;
                    var material = subMesh < sharedMaterials.Count
                        ? sharedMaterials[subMesh]
                        : null;
                    subMeshes[subMesh] = CreateSubMeshUpdate(
                        objectName,
                        material,
                        indexOffset,
                        indexCount,
                        uvs != null);
                }
                if (indices.Count == 0 || indices.Count % 3 != 0)
                {
                    NativeBridge.ReportErrorOnce(
                        $"Yutrel Mesh '{objectName}' has no valid triangle indices.");
                    return false;
                }
                update = NativeBridge.SceneMeshUpdate.AddOrReplace(
                    EntityId.ToULong(filter.GetEntityId()),
                    vertices,
                    normals,
                    uvs,
                    indices.ToArray(),
                    subMeshes,
                    localToWorld);
                return true;
            }

            private static NativeBridge.SceneSubMeshUpdate CreateSubMeshUpdate(
                string objectName,
                Material material,
                uint indexOffset,
                uint indexCount,
                bool hasUvs)
            {
                var color = Vector3.zero;
                var luminanceNits = 0.0f;
                var doubleSided = false;
                var textureEncoding = NativeBridge.ExternalTextureEncoding.LinearSrgb;
                Color[] pixels = null;
                uint textureWidth = 0;
                uint textureHeight = 0;
                var uvScale = Vector2.one;
                var uvOffset = Vector2.zero;
                var materialId = 0ul;
                var materialType = NativeBridge.SceneMaterialType.FallbackDiffuse;
                var openPbr = default(OpenPBRMaterialData);
                var alphaClip = default(OpenPBRAlphaClipData);

                if (material == null)
                {
                    NativeBridge.ReportInfoOnce(
                        $"Yutrel Mesh '{objectName}' has no Material; gray Diffuse fallback is used.");
                }
                else
                {
                    materialId = EntityId.ToULong(material.GetEntityId());
                    doubleSided = material.doubleSidedGI ||
                                  material.HasProperty(CullModeProperty) &&
                                  Mathf.RoundToInt(material.GetFloat(CullModeProperty)) ==
                                  OpenPBRMaterialAdapter.CullModeOffValueFor(material);
                    if (OpenPBRMaterialAdapter.TryRead(material, out openPbr))
                    {
                        materialType = NativeBridge.SceneMaterialType.OpenPBR;
                        OpenPBRMaterialAdapter.TryReadAlphaClip(material, out alphaClip);
                    }
                    else if (OpenPBRMaterialAdapter.IsOpenPBR(material))
                    {
                        NativeBridge.ReportErrorOnce(
                            $"Yutrel OpenPBR Material '{material.name}' has missing or invalid parameters; " +
                            "gray Diffuse fallback is used.");
                    }
                    else
                    {
                        var shaderName = material.shader != null
                            ? material.shader.name
                            : "<missing>";
                        NativeBridge.ReportInfoOnce(
                            $"Yutrel Material '{material.name}' uses unsupported Shader '{shaderName}'; " +
                            "gray Diffuse fallback is used.");
                    }
                }

                if (material != null && material.HasProperty(EmissiveLuminanceProperty))
                {
                    luminanceNits = material.GetFloat(EmissiveLuminanceProperty);
                    if (!IsFinite(luminanceNits) || luminanceNits < 0.0f)
                    {
                        NativeBridge.ReportErrorOnce(
                            $"Yutrel Material '{material.name}' has invalid emissive luminance.");
                        luminanceNits = 0.0f;
                    }
                    var materialColor = material.HasProperty(EmissiveColorProperty)
                        ? material.GetColor(EmissiveColorProperty).linear
                        : Color.white;
                    color = PhotometricColor.NormalizeLinearSrgb(materialColor);
                    var textureProperty = EmissiveTexturePropertyFor(material);
                    if (luminanceNits > 0.0f && textureProperty != null)
                    {
                        var texture = material.GetTexture(textureProperty);
                        if (texture != null && !hasUvs)
                        {
                            NativeBridge.ReportErrorOnce(
                                $"Yutrel Mesh '{objectName}' needs UVs for emissive texture '{texture.name}'. " +
                                "The constant emissive color will be used instead.");
                        }
                        else if (texture != null && TryReadTexture(
                                     objectName,
                                     texture,
                                     false,
                                     false,
                                     out pixels,
                                     out textureWidth,
                                     out textureHeight,
                                     out textureEncoding))
                        {
                            uvScale = material.GetTextureScale(textureProperty);
                            uvOffset = material.GetTextureOffset(textureProperty);
                        }
                    }
                }

                // --- OpenPBR texture slots (Sponza maps) ---
                var baseColorSlot = default(NativeBridge.OpenPBRTextureSlot);
                var normalSlot = default(NativeBridge.OpenPBRTextureSlot);
                var specularRoughnessSlot = default(NativeBridge.OpenPBRTextureSlot);
                var baseMetalnessSlot = default(NativeBridge.OpenPBRTextureSlot);
                var materialAoSlot = default(NativeBridge.OpenPBRTextureSlot);
                if (materialType == NativeBridge.SceneMaterialType.OpenPBR && material != null)
                {
                    foreach (var mapping in OpenPBRMaterialAdapter.GetTextureMappings(material))
                    {
                        if (!material.HasProperty(mapping.propertyName))
                        {
                            continue;
                        }
                        var texture = material.GetTexture(mapping.propertyName);
                        if (texture == null || !hasUvs)
                        {
                            continue;
                        }
                        if (!TryReadTextureCached(
                                objectName,
                                materialId,
                                mapping,
                                texture,
                                out var mappedPixels,
                                out var mappedWidth,
                                out var mappedHeight,
                                out var mappedEncoding))
                        {
                            continue; // error already reported; constant fallback is used
                        }
                        var slot = new NativeBridge.OpenPBRTextureSlot(
                            mappedEncoding,
                            mappedPixels,
                            mappedWidth,
                            mappedHeight,
                            material.GetTextureScale(mapping.propertyName),
                            material.GetTextureOffset(mapping.propertyName));
                        switch (mapping.slot)
                        {
                            case OpenPBRTextureSlot.BaseColor: baseColorSlot = slot; break;
                            case OpenPBRTextureSlot.Normal: normalSlot = slot; break;
                            case OpenPBRTextureSlot.SpecularRoughness: specularRoughnessSlot = slot; break;
                            case OpenPBRTextureSlot.BaseMetalness: baseMetalnessSlot = slot; break;
                            case OpenPBRTextureSlot.MaterialAo: materialAoSlot = slot; break;
                        }
                    }
                }

                return new NativeBridge.SceneSubMeshUpdate(
                    indexOffset,
                    indexCount,
                    color,
                    luminanceNits,
                    doubleSided,
                    textureEncoding,
                    pixels,
                    textureWidth,
                    textureHeight,
                    uvScale,
                    uvOffset,
                    materialId,
                    materialType,
                    openPbr,
                    baseColorSlot,
                    normalSlot,
                    specularRoughnessSlot,
                    baseMetalnessSlot,
                    materialAoSlot,
                    alphaClip);
            }

            private static string EmissiveTexturePropertyFor(Material material)
            {
                var shaderName = material.shader != null ? material.shader.name : string.Empty;
                if (shaderName == DefaultLitShaderName &&
                    material.HasProperty(UseEmissiveTextureProperty) &&
                    material.GetFloat(UseEmissiveTextureProperty) > 0.5f)
                {
                    return EmissiveTextureProperty;
                }
                return shaderName == UnlitShaderName ? MainTextureProperty : null;
            }

            // Per-material texture read cache, valid for one structure scan.
            // Keys are (materialId, propertyName); entries hold the decoded
            // float4 pixel array together with the Texture.updateCount used to
            // detect re-imports (so a modified texture invalidates the cache).
            private sealed class CachedTexture
            {
                internal readonly Color[] pixels;
                internal readonly uint width;
                internal readonly uint height;
                internal readonly NativeBridge.ExternalTextureEncoding encoding;
                internal readonly uint updateCount;

                internal CachedTexture(
                    Color[] pixels,
                    uint width,
                    uint height,
                    NativeBridge.ExternalTextureEncoding encoding,
                    uint updateCount)
                {
                    this.pixels = pixels;
                    this.width = width;
                    this.height = height;
                    this.encoding = encoding;
                    this.updateCount = updateCount;
                }
            }

            private static readonly Dictionary<ulong, Dictionary<string, CachedTexture>>
                TextureReadCache = new();

            private static bool TryReadTextureCached(
                string objectName,
                ulong materialId,
                OpenPBRTextureMapping mapping,
                Texture texture,
                out Color[] pixels,
                out uint width,
                out uint height,
                out NativeBridge.ExternalTextureEncoding encoding)
            {
                pixels = null;
                width = 0;
                height = 0;
                encoding = NativeBridge.ExternalTextureEncoding.LinearSrgb;
                if (TextureReadCache.TryGetValue(materialId, out var perMaterial) &&
                    perMaterial.TryGetValue(mapping.propertyName, out var cached) &&
                    cached.updateCount == texture.updateCount)
                {
                    pixels = cached.pixels;
                    width = cached.width;
                    height = cached.height;
                    encoding = cached.encoding;
                    return true;
                }
                if (!TryReadTexture(
                        objectName,
                        texture,
                        mapping.invert,
                        mapping.slot == OpenPBRTextureSlot.Normal,
                        out pixels,
                        out width,
                        out height,
                        out encoding))
                {
                    return false;
                }
                if (mapping.isSRGB)
                {
                    encoding = NativeBridge.ExternalTextureEncoding.Srgb;
                }
                if (perMaterial == null || !TextureReadCache.TryGetValue(materialId, out perMaterial))
                {
                    perMaterial = new Dictionary<string, CachedTexture>();
                    TextureReadCache[materialId] = perMaterial;
                }
                perMaterial[mapping.propertyName] =
                    new CachedTexture(pixels, width, height, encoding, texture.updateCount);
                return true;
            }

            private static bool TryReadTexture(
                string objectName,
                Texture texture,
                bool invert,
                bool unpackNormal,
                out Color[] pixels,
                out uint width,
                out uint height,
                out NativeBridge.ExternalTextureEncoding encoding)
            {
                pixels = null;
                width = 0;
                height = 0;
                encoding = NativeBridge.ExternalTextureEncoding.LinearSrgb;
                if (texture is not Texture2D texture2D || !texture2D.isReadable)
                {
                    NativeBridge.ReportErrorOnce(
                        $"Yutrel texture '{texture.name}' on Mesh '{objectName}' must be a readable Texture2D. " +
                        "The constant parameter will be used instead.");
                    return false;
                }

                try
                {
                    pixels = texture2D.GetPixels(0);
                }
                catch (UnityException exception)
                {
                    NativeBridge.ReportErrorOnce(
                        $"Yutrel could not read texture '{texture.name}': {exception.Message}");
                    pixels = null;
                    return false;
                }
                if (texture2D.width <= 0 || texture2D.height <= 0 ||
                    pixels.Length != texture2D.width * texture2D.height)
                {
                    pixels = null;
                    return false;
                }
                for (var pixelIndex = 0; pixelIndex < pixels.Length; pixelIndex++)
                {
                    var pixel = pixels[pixelIndex];
                    if (!IsFinite(pixel.r) || !IsFinite(pixel.g) ||
                        !IsFinite(pixel.b) || !IsFinite(pixel.a))
                    {
                        NativeBridge.ReportErrorOnce(
                            $"Yutrel texture '{texture.name}' contains non-finite pixels. " +
                            "The constant parameter will be used instead.");
                        pixels = null;
                        return false;
                    }
                    if (unpackNormal)
                    {
                        // Match Unity's UnpackNormalMapRGorAG. DXT5nm stores X
                        // in A with R=1; BC5 stores X in R with A=1, so R*A
                        // handles both desktop encodings. Upload canonical RGB
                        // so the native renderer remains Unity-independent.
                        var normalX = pixel.r * pixel.a * 2.0f - 1.0f;
                        var normalY = pixel.g * 2.0f - 1.0f;
                        var normalZ = Mathf.Sqrt(Mathf.Max(
                            1.0e-16f,
                            1.0f - Mathf.Clamp01(normalX * normalX + normalY * normalY)));
                        pixel = new Color(
                            normalX * 0.5f + 0.5f,
                            normalY * 0.5f + 0.5f,
                            normalZ * 0.5f + 0.5f,
                            1.0f);
                    }
                    else if (invert)
                    {
                        // TODO(临时): smoothness -> roughness 反相在主机侧逐像素做
                        // (1-x)。正解候选: a) 渲染器 Texture 层加 invert 变换;
                        // b) 上传原始 smoothness,OpenPBRSurface 对
                        // specular_roughness 槽采样后反相; c) 素材预处理导出
                        // roughness 贴图。
                        pixel = new Color(
                            1.0f - pixel.r,
                            1.0f - pixel.g,
                            1.0f - pixel.b,
                            1.0f - pixel.a);
                    }
                    pixels[pixelIndex] = new Color(
                        Mathf.Max(pixel.r, 0.0f),
                        Mathf.Max(pixel.g, 0.0f),
                        Mathf.Max(pixel.b, 0.0f),
                        Mathf.Max(pixel.a, 0.0f));
                }
                width = (uint)texture2D.width;
                height = (uint)texture2D.height;
                encoding = !unpackNormal && texture2D.isDataSRGB
                    ? NativeBridge.ExternalTextureEncoding.Srgb
                    : NativeBridge.ExternalTextureEncoding.LinearSrgb;
                return true;
            }

            private int ComputeMaterialSignature(Mesh mesh, MeshRenderer renderer)
            {
                unchecked
                {
                    var hash = mesh.subMeshCount;
                    sharedMaterials.Clear();
                    renderer.GetSharedMaterials(sharedMaterials);
                    for (var subMesh = 0; subMesh < mesh.subMeshCount; subMesh++)
                    {
                        var material = subMesh < sharedMaterials.Count
                            ? sharedMaterials[subMesh]
                            : null;
                        hash = hash * 397 ^ (material != null
                            ? EntityId.ToULong(material.GetEntityId()).GetHashCode()
                            : 0);
                        if (material == null)
                        {
                            continue;
                        }
                        hash = hash * 397 ^ (material.shader != null ? material.shader.name.GetHashCode() : 0);
                        hash = hash * 397 ^ material.doubleSidedGI.GetHashCode();
                        if (material.HasProperty(EmissiveLuminanceProperty))
                        {
                            hash = hash * 397 ^ material.GetFloat(EmissiveLuminanceProperty).GetHashCode();
                        }
                        if (material.HasProperty(EmissiveColorProperty))
                        {
                            hash = hash * 397 ^ material.GetColor(EmissiveColorProperty).GetHashCode();
                        }
                        if (material.HasProperty(CullModeProperty))
                        {
                            hash = hash * 397 ^ material.GetFloat(CullModeProperty).GetHashCode();
                        }
                        if (OpenPBRMaterialAdapter.IsOpenPBR(material))
                        {
                            hash = hash * 397 ^ OpenPBRMaterialAdapter.ComputeSignature(material);
                            hash = HashOpenPBRTextures(material, hash);
                            if (OpenPBRMaterialAdapter.TryReadAlphaClip(material, out var alphaClip))
                            {
                                hash = HashAlphaClip(hash, alphaClip);
                            }
                        }
                        var textureProperty = EmissiveTexturePropertyFor(material);
                        if (textureProperty == null)
                        {
                            continue;
                        }
                        var texture = material.GetTexture(textureProperty);
                        hash = hash * 397 ^ (texture != null
                            ? EntityId.ToULong(texture.GetEntityId()).GetHashCode()
                            : 0);
                        if (texture != null)
                        {
                            hash = hash * 397 ^ texture.updateCount.GetHashCode();
                            hash = hash * 397 ^ material.GetTextureScale(textureProperty).GetHashCode();
                            hash = hash * 397 ^ material.GetTextureOffset(textureProperty).GetHashCode();
                        }
                    }
                    return hash;
                }
            }

            private static int HashOpenPBRTextures(Material material, int hash)
            {
                unchecked
                {
                    foreach (var mapping in OpenPBRMaterialAdapter.GetTextureMappings(material))
                    {
                        if (!material.HasProperty(mapping.propertyName))
                        {
                            continue;
                        }
                        var texture = material.GetTexture(mapping.propertyName);
                        hash = hash * 397 ^ (texture != null
                            ? EntityId.ToULong(texture.GetEntityId()).GetHashCode()
                            : 0);
                        if (texture != null)
                        {
                            hash = hash * 397 ^ texture.updateCount.GetHashCode();
                            hash = hash * 397 ^
                                   material.GetTextureScale(mapping.propertyName).GetHashCode();
                            hash = hash * 397 ^
                                   material.GetTextureOffset(mapping.propertyName).GetHashCode();
                        }
                    }
                }
                return hash;
            }

            private static int HashAlphaClip(int hash, OpenPBRAlphaClipData alphaClip)
            {
                unchecked
                {
                    hash = hash * 397 ^ (alphaClip.enabled ? 1 : 0);
                    hash = hash * 397 ^ alphaClip.cutoff.GetHashCode();
                    hash = hash * 397 ^ alphaClip.baseColorAlpha.GetHashCode();
                }
                return hash;
            }

            private static bool TryReadSubMeshMaterialState(
                Material material,
                out SubMeshMaterialState state)
            {
                state = default;
                var materialId = material != null
                    ? EntityId.ToULong(material.GetEntityId())
                    : 0ul;
                var doubleSided = material != null &&
                                  (material.doubleSidedGI ||
                                   material.HasProperty(CullModeProperty) &&
                                   Mathf.RoundToInt(material.GetFloat(CullModeProperty)) ==
                                   OpenPBRMaterialAdapter.CullModeOffValueFor(material));
                var isOpenPbr = OpenPBRMaterialAdapter.IsOpenPBR(material);
                var materialType = isOpenPbr
                    ? NativeBridge.SceneMaterialType.OpenPBR
                    : NativeBridge.SceneMaterialType.FallbackDiffuse;
                var paramsHash = 0;
                if (isOpenPbr && OpenPBRMaterialAdapter.TryRead(material, out _))
                {
                    paramsHash = OpenPBRMaterialAdapter.ComputeSignature(material);
                }
                unchecked
                {
                    var restHash = (int)materialId.GetHashCode();
                    restHash = restHash * 397 ^ (int)materialType;
                    restHash = restHash * 397 ^ (doubleSided ? 1 : 0);
                    if (material != null)
                    {
                        if (material.HasProperty(EmissiveLuminanceProperty))
                        {
                            restHash = restHash * 397 ^
                                       material.GetFloat(EmissiveLuminanceProperty).GetHashCode();
                        }
                        if (material.HasProperty(EmissiveColorProperty))
                        {
                            restHash = restHash * 397 ^
                                       material.GetColor(EmissiveColorProperty).GetHashCode();
                        }
                        if (material.HasProperty(CullModeProperty))
                        {
                            restHash = restHash * 397 ^
                                       material.GetFloat(CullModeProperty).GetHashCode();
                        }
                        var textureProperty = EmissiveTexturePropertyFor(material);
                        if (textureProperty != null)
                        {
                            var texture = material.GetTexture(textureProperty);
                            restHash = restHash * 397 ^ (texture != null
                                ? EntityId.ToULong(texture.GetEntityId()).GetHashCode()
                                : 0);
                            if (texture != null)
                            {
                                restHash = restHash * 397 ^ texture.updateCount.GetHashCode();
                                restHash = restHash * 397 ^
                                           material.GetTextureScale(textureProperty).GetHashCode();
                                restHash = restHash * 397 ^
                                           material.GetTextureOffset(textureProperty).GetHashCode();
                            }
                        }
                        if (isOpenPbr)
                        {
                            restHash = HashOpenPBRTextures(material, restHash);
                            if (OpenPBRMaterialAdapter.TryReadAlphaClip(material, out var alphaClip))
                            {
                                restHash = HashAlphaClip(restHash, alphaClip);
                            }
                        }
                    }
                    state = new SubMeshMaterialState
                    {
                        materialId = materialId,
                        materialType = materialType,
                        doubleSided = doubleSided,
                        restHash = restHash,
                        paramsHash = paramsHash
                    };
                }
                return isOpenPbr;
            }

            private SubMeshMaterialState[] ComputeSubMeshMaterialStates(MeshFilter filter)
            {
                sharedMaterials.Clear();
                filter.GetComponent<MeshRenderer>().GetSharedMaterials(sharedMaterials);
                var states = new SubMeshMaterialState[filter.sharedMesh.subMeshCount];
                for (var subMesh = 0; subMesh < states.Length; subMesh++)
                {
                    var material = subMesh < sharedMaterials.Count
                        ? sharedMaterials[subMesh]
                        : null;
                    TryReadSubMeshMaterialState(material, out states[subMesh]);
                }
                return states;
            }

            // Returns true and fills `updates` when the only change vs. the
            // tracked snapshot is OpenPBR parameter values on (a subset of)
            // submeshes. Any change to geometry identity, material identity,
            // material type, double-sidedness, emissive or texture properties
            // falls back to a full mesh update.
            private bool TryBuildMaterialUpdates(
                MeshFilter filter,
                ulong meshId,
                TrackedMesh tracked,
                out List<NativeBridge.SceneMaterialUpdate> updates)
            {
                updates = null;
                var states = ComputeSubMeshMaterialStates(filter);
                if (tracked.subMeshStates == null ||
                    states.Length != tracked.subMeshStates.Length)
                {
                    return false;
                }
                for (var i = 0; i < states.Length; i++)
                {
                    var previous = tracked.subMeshStates[i];
                    var current = states[i];
                    if (previous.restHash != current.restHash ||
                        previous.materialId != current.materialId ||
                        previous.materialType != current.materialType ||
                        previous.doubleSided != current.doubleSided)
                    {
                        return false;
                    }
                }
                var anyParamChanged = false;
                for (var i = 0; i < states.Length; i++)
                {
                    if (tracked.subMeshStates[i].paramsHash != states[i].paramsHash)
                    {
                        anyParamChanged = true;
                        break;
                    }
                }
                if (!anyParamChanged)
                {
                    return false;
                }

                sharedMaterials.Clear();
                filter.GetComponent<MeshRenderer>().GetSharedMaterials(sharedMaterials);
                updates = new List<NativeBridge.SceneMaterialUpdate>();
                for (var i = 0; i < states.Length; i++)
                {
                    if (states[i].materialType != NativeBridge.SceneMaterialType.OpenPBR)
                    {
                        continue;
                    }
                    var material = i < sharedMaterials.Count ? sharedMaterials[i] : null;
                    if (material == null ||
                        !OpenPBRMaterialAdapter.TryRead(material, out var data))
                    {
                        updates = null;
                        return false;
                    }
                    updates.Add(new NativeBridge.SceneMaterialUpdate(
                        meshId,
                        (uint)i,
                        states[i].materialId,
                        states[i].doubleSided,
                        data));
                }
                if (updates.Count == 0)
                {
                    updates = null;
                    return false;
                }
                tracked.subMeshStates = states;
                return true;
            }

            private static bool IsValidTransform(Matrix4x4 matrix)
            {
                for (var column = 0; column < 4; column++)
                {
                    for (var row = 0; row < 4; row++)
                    {
                        if (!IsFinite(matrix[row, column]))
                        {
                            return false;
                        }
                    }
                }
                if (Mathf.Abs(matrix[3, 0]) > CameraChangeEpsilon ||
                    Mathf.Abs(matrix[3, 1]) > CameraChangeEpsilon ||
                    Mathf.Abs(matrix[3, 2]) > CameraChangeEpsilon ||
                    Mathf.Abs(matrix[3, 3] - 1.0f) > CameraChangeEpsilon)
                {
                    return false;
                }
                var x = (Vector3)matrix.GetColumn(0);
                var y = (Vector3)matrix.GetColumn(1);
                var z = (Vector3)matrix.GetColumn(2);
                return x.sqrMagnitude >= 1e-12f &&
                       y.sqrMagnitude >= 1e-12f &&
                       z.sqrMagnitude >= 1e-12f &&
                       Mathf.Abs(Vector3.Dot(x, Vector3.Cross(y, z))) >= 1e-8f;
            }

            private static bool IsFinite(Vector3 value)
            {
                return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
            }

            private static bool IsFinite(float value)
            {
                return !float.IsNaN(value) && !float.IsInfinity(value);
            }

            private static bool LightNear(
                NativeBridge.DirectionalLightUpdate lhs,
                NativeBridge.DirectionalLightUpdate rhs)
            {
                return lhs.enabled == rhs.enabled &&
                       Mathf.Abs(lhs.colorLinearSrgb.x - rhs.colorLinearSrgb.x) <= CameraChangeEpsilon &&
                       Mathf.Abs(lhs.colorLinearSrgb.y - rhs.colorLinearSrgb.y) <= CameraChangeEpsilon &&
                       Mathf.Abs(lhs.colorLinearSrgb.z - rhs.colorLinearSrgb.z) <= CameraChangeEpsilon &&
                       Mathf.Abs(lhs.illuminanceLux - rhs.illuminanceLux) <= CameraChangeEpsilon &&
                       (lhs.direction - rhs.direction).sqrMagnitude <=
                       CameraChangeEpsilon * CameraChangeEpsilon;
            }

            private static bool EnvironmentNear(
                NativeBridge.EnvironmentUpdate lhs,
                NativeBridge.EnvironmentUpdate rhs)
            {
                return lhs.enabled == rhs.enabled &&
                       (!lhs.enabled ||
                        (string.Equals(
                             lhs.hdrPath,
                             rhs.hdrPath,
                             StringComparison.OrdinalIgnoreCase) &&
                         Mathf.Abs(lhs.intensity - rhs.intensity) <= CameraChangeEpsilon));
            }

#if UNITY_EDITOR
            private void OnObjectChangesPublished(
                ref UnityEditor.ObjectChangeEventStream stream)
            {
                // Event-driven fast path (like Blender's depsgraph tagging):
                // schedule a re-scan as soon as anything the path tracer reads
                // changes. Only kinds that can affect the rendered scene are
                // handled; irrelevant events (e.g. pure scene transitions) are
                // ignored because the periodic scan below is the safety net
                // for anything this filter misses.
                for (var i = 0; i < stream.length; i++)
                {
                    switch (stream.GetEventType(i))
                    {
                        // Material/Texture/Shader/Mesh asset edits, object
                        // structure and component property changes all affect
                        // the captured scene.
                        case UnityEditor.ObjectChangeKind.ChangeAssetObjectProperties:
                        case UnityEditor.ObjectChangeKind.CreateAssetObject:
                        case UnityEditor.ObjectChangeKind.DestroyAssetObject:
                        case UnityEditor.ObjectChangeKind.ChangeGameObjectOrComponentProperties:
                        case UnityEditor.ObjectChangeKind.CreateGameObjectHierarchy:
                        case UnityEditor.ObjectChangeKind.DestroyGameObjectHierarchy:
                        case UnityEditor.ObjectChangeKind.ChangeGameObjectStructure:
                        case UnityEditor.ObjectChangeKind.ChangeGameObjectStructureHierarchy:
                        case UnityEditor.ObjectChangeKind.ChangeGameObjectParent:
                        case UnityEditor.ObjectChangeKind.ChangeChildrenOrder:
                        case UnityEditor.ObjectChangeKind.ChangeRootOrder:
                        case UnityEditor.ObjectChangeKind.UpdatePrefabInstances:
                            structureDirty = true;
                            editorStructureScanNotBefore =
                                Time.realtimeSinceStartupAsDouble + EditorChangeDebounce;
                            return;
                        default:
                            continue;
                    }
                }
            }

            private void OnHierarchyChanged()
            {
                structureDirty = true;
            }
#endif
        }

#if UNITY_EDITOR
        private sealed class SceneViewRepaintScheduler
        {
            private const double RepaintIntervalSeconds = 1.0 / 60.0;

            private bool scheduled;
            private double nextRepaintTime;

            internal void Request()
            {
                if (scheduled)
                {
                    return;
                }

                scheduled = true;
                UnityEditor.EditorApplication.update += OnEditorUpdate;
            }

            internal void Dispose()
            {
                if (scheduled)
                {
                    UnityEditor.EditorApplication.update -= OnEditorUpdate;
                    scheduled = false;
                }
            }

            private void OnEditorUpdate()
            {
                var now = UnityEditor.EditorApplication.timeSinceStartup;
                if (now < nextRepaintTime)
                {
                    return;
                }

                UnityEditor.EditorApplication.update -= OnEditorUpdate;
                scheduled = false;
                nextRepaintTime = now + RepaintIntervalSeconds;
                UnityEditor.EditorApplication.QueuePlayerLoopUpdate();
                UnityEditor.SceneView.RepaintAll();
            }
        }
#endif

        private const float CameraChangeEpsilon = 1e-6f;

        private readonly Dictionary<ViewKey, ViewState> views = new();
        private readonly SceneChangeTracker sceneChangeTracker;
#if UNITY_EDITOR
        private readonly SceneViewRepaintScheduler sceneViewRepaintScheduler = new();
#endif
        private readonly bool nativeAcquired;
        private uint nextNativeViewId = 1u;

        internal YutrelPathTracer()
        {
            nativeAcquired = NativeBridge.Acquire();
            sceneChangeTracker = nativeAcquired ? new SceneChangeTracker() : null;
        }

        protected override YutrelRendererOutput RecordScene(
            RenderGraph renderGraph,
            in YutrelCameraRenderContext context)
        {
            if (!nativeAcquired)
            {
                return CreateBlackFallback(renderGraph, context.targetSize, context.sceneColorFormat);
            }

            var camera = context.camera;
            var isSceneView = false;
            switch (camera.cameraType)
            {
                case CameraType.Game:
                    break;
#if UNITY_EDITOR
                case CameraType.SceneView:
                    isSceneView = true;
                    break;
                case CameraType.Preview:
                    break;
#endif
                default:
                    NativeBridge.ReportInfoOnce(
                        "YutrelPathTracer stage 1B only renders Game, SceneView, and Preview Cameras.");
                    return CreateBlackFallback(renderGraph, context.targetSize, context.sceneColorFormat);
            }

            var viewLabel = camera.cameraType.ToString();
            if (camera.orthographic)
            {
                NativeBridge.ReportInfoOnce(
                    "YutrelPathTracer stage 1B requires perspective Game, SceneView, and Preview Cameras.");
                return CreateBlackFallback(renderGraph, context.targetSize, context.sceneColorFormat);
            }

            var view = GetOrCreateView(camera);

            if (sceneChangeTracker == null || !sceneChangeTracker.Synchronize(camera))
            {
                return CreateBlackFallback(renderGraph, context.targetSize, context.sceneColorFormat);
            }

            var resized = view.resources.Ensure(camera, context.targetSize);
            var renderTexture = view.resources.color?.rt;
            if (renderTexture == null || !renderTexture.IsCreated())
            {
                NativeBridge.ReportErrorOnce(
                    $"YutrelPathTracer failed to create its {viewLabel} Camera RenderTexture.");
                return CreateBlackFallback(renderGraph, context.targetSize, context.sceneColorFormat);
            }

            var outputPointer = renderTexture.GetNativeTexturePtr();
            if (outputPointer == IntPtr.Zero)
            {
                NativeBridge.ReportErrorOnce("YutrelPathTracer received a null native texture pointer.");
                return CreateBlackFallback(renderGraph, context.targetSize, context.sceneColorFormat);
            }

            var cameraToWorld = BuildCameraToWorld(camera.transform);
            var cameraChanged = !view.previousCameraValid ||
                                !MatrixNear(view.previousCameraToWorld, cameraToWorld) ||
                                Mathf.Abs(view.previousVerticalFov - camera.fieldOfView) > CameraChangeEpsilon;
            var resetAccumulation = resized || cameraChanged;
            view.previousCameraToWorld = cameraToWorld;
            view.previousVerticalFov = camera.fieldOfView;
            view.previousCameraValid = true;

            var output = renderGraph.ImportTexture(
                view.resources.color,
                new ImportResourceParams
                {
                    clearOnFirstUse = true,
                    clearColor = Color.black,
                    discardOnLastUse = false
                });
            var flipOutputY = SystemInfo.graphicsUVStartsAtTop;
            PathTracePass.Record(
                renderGraph,
                output,
                outputPointer,
                context.targetSize,
                view.nativeViewId,
                flipOutputY,
                cameraToWorld,
                camera.fieldOfView,
                context.preExposure,
                resetAccumulation);
#if UNITY_EDITOR
            if (isSceneView)
            {
                sceneViewRepaintScheduler.Request();
            }
#endif
            return new YutrelRendererOutput(output);
        }

        protected override void Dispose(bool disposing)
        {
            foreach (var view in views.Values)
            {
                view.Release();
            }
            views.Clear();
            sceneChangeTracker?.Dispose();
#if UNITY_EDITOR
            sceneViewRepaintScheduler.Dispose();
#endif
            nextNativeViewId = 1u;

            if (nativeAcquired)
            {
                NativeBridge.Release();
            }
        }

        private ViewState GetOrCreateView(Camera camera)
        {
            var key = new ViewKey(camera);
            if (views.TryGetValue(key, out var view))
            {
                return view;
            }

            view = new ViewState(nextNativeViewId++);
            views.Add(key, view);
            return view;
        }

        private static Matrix4x4 BuildCameraToWorld(Transform transform)
        {
            var matrix = Matrix4x4.identity;
            var right = transform.right;
            var up = transform.up;
            var backward = -transform.forward;
            var position = transform.position;
            matrix.SetColumn(0, new Vector4(right.x, right.y, right.z, 0.0f));
            matrix.SetColumn(1, new Vector4(up.x, up.y, up.z, 0.0f));
            matrix.SetColumn(2, new Vector4(backward.x, backward.y, backward.z, 0.0f));
            matrix.SetColumn(3, new Vector4(position.x, position.y, position.z, 1.0f));
            return matrix;
        }

        private static bool MatrixNear(Matrix4x4 lhs, Matrix4x4 rhs)
        {
            for (var column = 0; column < 4; column++)
            {
                for (var row = 0; row < 4; row++)
                {
                    if (Mathf.Abs(lhs[row, column] - rhs[row, column]) > CameraChangeEpsilon)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        private static YutrelRendererOutput CreateBlackFallback(
            RenderGraph renderGraph,
            Vector2Int targetSize,
            GraphicsFormat format)
        {
            var output = renderGraph.CreateTexture(new TextureDesc(targetSize.x, targetSize.y)
            {
                colorFormat = format,
                clearBuffer = true,
                clearColor = Color.black,
                name = "YutrelPathTracer Unavailable"
            });
            return new YutrelRendererOutput(output);
        }
    }
}

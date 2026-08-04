using System;
using System.Collections.Generic;
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
            private const double RuntimeStructureScanInterval = 0.25;
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
                internal Matrix4x4 localToWorld;
            }

            private readonly Dictionary<ulong, TrackedMesh> meshes = new();
            private readonly HashSet<ulong> seenMeshIds = new();
            private readonly List<ulong> removedMeshIds = new();
            private readonly List<NativeBridge.SceneMeshUpdate> pendingUpdates = new();
            private readonly List<Material> sharedMaterials = new();

            private Light selectedLight;
            private NativeBridge.DirectionalLightUpdate previousLight;
            private bool previousLightValid;
            private bool initialized;
            private bool structureDirty = true;
            private double nextRuntimeStructureScan;
            private double editorStructureScanNotBefore;
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
            }

            internal bool Synchronize()
            {
                pendingUpdates.Clear();
                var now = Time.realtimeSinceStartupAsDouble;
                var runtimeScanDue = Application.isPlaying && now >= nextRuntimeStructureScan;
                var structureScanReady = !initialized || now >= editorStructureScanNotBefore;
                if (structureScanReady && (!initialized || structureDirty || runtimeScanDue))
                {
                    ScanStructure();
                    initialized = true;
                    structureDirty = false;
                    nextRuntimeStructureScan = now + RuntimeStructureScanInterval;
                }

                CompareTransforms();
                var light = CurrentLightState();
                var lightChanged = !previousLightValid || !LightNear(previousLight, light);
                if (pendingUpdates.Count == 0 && !lightChanged)
                {
                    return true;
                }

                revision++;
                if (!NativeBridge.SubmitSceneDelta(
                        pendingUpdates,
                        revision,
                        lightChanged ? light : null))
                {
                    meshes.Clear();
                    selectedLight = null;
                    previousLightValid = false;
                    initialized = false;
                    structureDirty = true;
                    return false;
                }
                previousLight = light;
                previousLightValid = true;
                return true;
            }

            private void ScanStructure()
            {
                seenMeshIds.Clear();
                foreach (var filter in UnityEngine.Object.FindObjectsByType<MeshFilter>(
                             FindObjectsInactive.Exclude))
                {
                    var meshRenderer = filter.GetComponent<MeshRenderer>();
                    if (meshRenderer == null || !meshRenderer.enabled ||
                        !filter.gameObject.activeInHierarchy || filter.sharedMesh == null)
                    {
                        continue;
                    }

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

                    if (!TryCreateMeshUpdate(filter, out var update))
                    {
                        continue;
                    }
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
                                  (int)CullMode.Off;
                    if (OpenPBRMaterialAdapter.TryRead(material, out openPbr))
                    {
                        materialType = NativeBridge.SceneMaterialType.OpenPBR;
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
                    openPbr);
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

            private static bool TryReadTexture(
                string objectName,
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
                if (texture is not Texture2D texture2D || !texture2D.isReadable)
                {
                    NativeBridge.ReportErrorOnce(
                        $"Yutrel emissive texture '{texture.name}' on Mesh '{objectName}' must be a readable Texture2D. " +
                        "The constant emissive color will be used instead.");
                    return false;
                }

                try
                {
                    pixels = texture2D.GetPixels(0);
                }
                catch (UnityException exception)
                {
                    NativeBridge.ReportErrorOnce(
                        $"Yutrel could not read emissive texture '{texture.name}': {exception.Message}");
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
                            $"Yutrel emissive texture '{texture.name}' contains non-finite pixels. " +
                            "The constant emissive color will be used instead.");
                        pixels = null;
                        return false;
                    }
                    pixels[pixelIndex] = new Color(
                        Mathf.Max(pixel.r, 0.0f),
                        Mathf.Max(pixel.g, 0.0f),
                        Mathf.Max(pixel.b, 0.0f),
                        Mathf.Max(pixel.a, 0.0f));
                }
                width = (uint)texture2D.width;
                height = (uint)texture2D.height;
                encoding = texture2D.isDataSRGB
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

#if UNITY_EDITOR
            private void OnObjectChangesPublished(
                ref UnityEditor.ObjectChangeEventStream stream)
            {
                structureDirty = true;
                editorStructureScanNotBefore =
                    Time.realtimeSinceStartupAsDouble + EditorChangeDebounce;
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

            if (sceneChangeTracker == null || !sceneChangeTracker.Synchronize())
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

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using YutrelRP;

namespace Yutrel.OfflineRenderer
{
    internal sealed class YutrelOfflineRenderer : YutrelRenderer
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
                    name: $"Yutrel Offline Color ({camera.name})");
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

            private sealed class TrackedMesh
            {
                internal MeshFilter filter;
                internal MeshRenderer renderer;
                internal ulong sharedMeshId;
                internal Matrix4x4 localToWorld;
            }

            private readonly Dictionary<ulong, TrackedMesh> meshes = new();
            private readonly HashSet<ulong> seenMeshIds = new();
            private readonly List<ulong> removedMeshIds = new();
            private readonly List<NativeBridge.SceneMeshUpdate> pendingUpdates = new();

            private Light selectedLight;
            private NativeBridge.DirectionalLightUpdate previousLight;
            private bool previousLightValid;
            private bool initialized;
            private bool structureDirty = true;
            private double nextRuntimeStructureScan;
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
                if (!initialized || structureDirty || runtimeScanDue)
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
                    if (meshes.TryGetValue(objectId, out var tracked) &&
                        tracked.sharedMeshId == sharedMeshId && filter.sharedMesh.isReadable &&
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
                            ? "Yutrel Offline Renderer lighting is disabled until one Directional Light is enabled."
                            : "Yutrel Offline Renderer lighting is disabled while multiple Directional Lights are enabled.");
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
                        Color.black,
                        0.0f,
                        Vector3.forward,
                        false);
                }
                if (!selectedLight.enabled || !selectedLight.gameObject.activeInHierarchy ||
                    selectedLight.type != LightType.Directional)
                {
                    structureDirty = true;
                    return new NativeBridge.DirectionalLightUpdate(
                        Color.black,
                        0.0f,
                        Vector3.forward,
                        false);
                }

                var color = selectedLight.color.linear;
                var intensity = selectedLight.intensity;
                var direction = -selectedLight.transform.forward;
                if (!IsFinite(color.r) || !IsFinite(color.g) || !IsFinite(color.b) ||
                    color.r < 0.0f || color.g < 0.0f || color.b < 0.0f ||
                    !IsFinite(intensity) || intensity < 0.0f ||
                    !IsFinite(direction) || direction.sqrMagnitude < 1e-12f)
                {
                    NativeBridge.ReportErrorOnce("The Yutrel Directional Light state is invalid.");
                    return new NativeBridge.DirectionalLightUpdate(
                        Color.black,
                        0.0f,
                        Vector3.forward,
                        false);
                }
                direction.Normalize();
                return new NativeBridge.DirectionalLightUpdate(
                    color,
                    intensity,
                    direction,
                    true);
            }

            private static bool TryCreateMeshUpdate(
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

                var indices = new List<uint>();
                for (var subMesh = 0; subMesh < mesh.subMeshCount; subMesh++)
                {
                    if (mesh.GetTopology(subMesh) != MeshTopology.Triangles)
                    {
                        NativeBridge.ReportErrorOnce(
                            $"Every submesh of Yutrel Mesh '{objectName}' must use Triangle topology.");
                        return false;
                    }
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
                    indices.ToArray(),
                    localToWorld);
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
                       Mathf.Abs(lhs.color.r - rhs.color.r) <= CameraChangeEpsilon &&
                       Mathf.Abs(lhs.color.g - rhs.color.g) <= CameraChangeEpsilon &&
                       Mathf.Abs(lhs.color.b - rhs.color.b) <= CameraChangeEpsilon &&
                       Mathf.Abs(lhs.intensity - rhs.intensity) <= CameraChangeEpsilon &&
                       (lhs.direction - rhs.direction).sqrMagnitude <=
                       CameraChangeEpsilon * CameraChangeEpsilon;
            }

#if UNITY_EDITOR
            private void OnObjectChangesPublished(
                ref UnityEditor.ObjectChangeEventStream stream)
            {
                structureDirty = true;
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

        internal YutrelOfflineRenderer()
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
                        "Yutrel Offline Renderer stage 1B only renders Game, SceneView, and Preview Cameras.");
                    return CreateBlackFallback(renderGraph, context.targetSize, context.sceneColorFormat);
            }

            var viewLabel = camera.cameraType.ToString();
            if (camera.orthographic)
            {
                NativeBridge.ReportInfoOnce(
                    "Yutrel Offline Renderer stage 1B requires perspective Game, SceneView, and Preview Cameras.");
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
                    $"Yutrel Offline Renderer failed to create its {viewLabel} Camera RenderTexture.");
                return CreateBlackFallback(renderGraph, context.targetSize, context.sceneColorFormat);
            }

            var outputPointer = renderTexture.GetNativeTexturePtr();
            if (outputPointer == IntPtr.Zero)
            {
                NativeBridge.ReportErrorOnce("Yutrel Offline Renderer received a null native texture pointer.");
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
            OfflinePathTracePass.Record(
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
                name = "Yutrel Offline Unavailable"
            });
            return new YutrelRendererOutput(output);
        }
    }
}

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
        private enum SceneUploadState
        {
            NotAttempted,
            Uploaded,
            Failed
        }

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
            private readonly int cameraId;
            private readonly CameraType cameraType;
            private readonly bool outputsToBackbuffer;

            internal ViewKey(Camera camera)
            {
                cameraId = camera.GetEntityId().GetHashCode();
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
                    var hash = cameraId;
                    hash = hash * 397 ^ (int)cameraType;
                    hash = hash * 397 ^ (outputsToBackbuffer ? 1 : 0);
                    return hash;
                }
            }
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
#if UNITY_EDITOR
        private readonly SceneViewRepaintScheduler sceneViewRepaintScheduler = new();
#endif
        private readonly bool nativeAcquired;
        private SceneUploadState sceneUploadState;
        private uint nextNativeViewId = 1u;

        internal YutrelOfflineRenderer()
        {
            nativeAcquired = NativeBridge.Acquire();
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

            if (sceneUploadState == SceneUploadState.NotAttempted)
            {
                sceneUploadState = TryUploadStaticScene()
                    ? SceneUploadState.Uploaded
                    : SceneUploadState.Failed;
            }
            if (sceneUploadState != SceneUploadState.Uploaded)
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
#if UNITY_EDITOR
            sceneViewRepaintScheduler.Dispose();
#endif
            sceneUploadState = SceneUploadState.NotAttempted;
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

        private static bool TryUploadStaticScene()
        {
            MeshFilter selectedFilter = null;
            MeshRenderer selectedRenderer = null;
            var meshCount = 0;
            foreach (var filter in UnityEngine.Object.FindObjectsByType<MeshFilter>(
                         FindObjectsInactive.Exclude))
            {
                var meshRenderer = filter.GetComponent<MeshRenderer>();
                if (meshRenderer == null || !meshRenderer.enabled || !filter.gameObject.activeInHierarchy)
                {
                    continue;
                }
                selectedFilter = filter;
                selectedRenderer = meshRenderer;
                meshCount++;
            }
            if (meshCount != 1 || selectedFilter == null || selectedRenderer == null)
            {
                NativeBridge.ReportErrorOnce(
                    $"Yutrel Offline Renderer stage 1B requires exactly one enabled MeshFilter + MeshRenderer; found {meshCount}.");
                return false;
            }

            Light selectedLight = null;
            var lightCount = 0;
            foreach (var light in UnityEngine.Object.FindObjectsByType<Light>(
                         FindObjectsInactive.Exclude))
            {
                if (!light.enabled || !light.gameObject.activeInHierarchy || light.type != LightType.Directional)
                {
                    continue;
                }
                selectedLight = light;
                lightCount++;
            }
            if (lightCount != 1 || selectedLight == null)
            {
                NativeBridge.ReportErrorOnce(
                    $"Yutrel Offline Renderer stage 1B requires exactly one enabled Directional Light; found {lightCount}.");
                return false;
            }

            var mesh = selectedFilter.sharedMesh;
            if (mesh == null || !mesh.isReadable)
            {
                NativeBridge.ReportErrorOnce("The stage 1B Mesh must exist and have Read/Write enabled.");
                return false;
            }
            var vertices = mesh.vertices;
            var normals = mesh.normals;
            if (vertices.Length == 0 || normals.Length != vertices.Length)
            {
                NativeBridge.ReportErrorOnce("The stage 1B Mesh must provide one normal per vertex.");
                return false;
            }

            var indices = new List<uint>();
            for (var subMesh = 0; subMesh < mesh.subMeshCount; subMesh++)
            {
                if (mesh.GetTopology(subMesh) != MeshTopology.Triangles)
                {
                    NativeBridge.ReportErrorOnce("Every stage 1B Mesh submesh must use Triangle topology.");
                    return false;
                }
                foreach (var index in mesh.GetIndices(subMesh))
                {
                    indices.Add((uint)index);
                }
            }
            if (indices.Count == 0 || indices.Count % 3 != 0)
            {
                NativeBridge.ReportErrorOnce("The stage 1B Mesh has no valid triangle indices.");
                return false;
            }

            var lightDirection = -selectedLight.transform.forward;
            if (lightDirection.sqrMagnitude < 1e-12f)
            {
                NativeBridge.ReportErrorOnce("The stage 1B Directional Light direction is invalid.");
                return false;
            }
            lightDirection.Normalize();
            return NativeBridge.SetStaticScene(
                vertices,
                normals,
                indices.ToArray(),
                selectedFilter.transform.localToWorldMatrix,
                selectedLight.color.linear,
                selectedLight.intensity,
                lightDirection);
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

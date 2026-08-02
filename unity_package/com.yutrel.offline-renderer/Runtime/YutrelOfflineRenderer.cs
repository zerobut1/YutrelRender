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

        private const float CameraChangeEpsilon = 1e-6f;

        private readonly CameraResources cameraResources = new();
        private readonly bool nativeAcquired;
        private SceneUploadState sceneUploadState;
        private int lockedCameraId;
        private bool cameraLocked;
        private Matrix4x4 previousCameraToWorld;
        private float previousVerticalFov;
        private bool previousCameraValid;

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
            if (camera.cameraType != CameraType.Game)
            {
                NativeBridge.ReportInfoOnce("Yutrel Offline Renderer stage 1A only renders one Game Camera.");
                return CreateBlackFallback(renderGraph, context.targetSize, context.sceneColorFormat);
            }
            if (camera.orthographic)
            {
                NativeBridge.ReportErrorOnce("Yutrel Offline Renderer stage 1A requires a perspective Camera.");
                return CreateBlackFallback(renderGraph, context.targetSize, context.sceneColorFormat);
            }

            var cameraId = camera.GetEntityId().GetHashCode();
            if (!cameraLocked)
            {
                cameraLocked = true;
                lockedCameraId = cameraId;
            }
            else if (lockedCameraId != cameraId)
            {
                NativeBridge.ReportInfoOnce("Yutrel Offline Renderer ignored an additional Game Camera.");
                return CreateBlackFallback(renderGraph, context.targetSize, context.sceneColorFormat);
            }

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

            var resized = cameraResources.Ensure(camera, context.targetSize);
            var renderTexture = cameraResources.color?.rt;
            if (renderTexture == null || !renderTexture.IsCreated())
            {
                NativeBridge.ReportErrorOnce("Yutrel Offline Renderer failed to create its Game Camera RenderTexture.");
                return CreateBlackFallback(renderGraph, context.targetSize, context.sceneColorFormat);
            }

            var outputPointer = renderTexture.GetNativeTexturePtr();
            if (outputPointer == IntPtr.Zero)
            {
                NativeBridge.ReportErrorOnce("Yutrel Offline Renderer received a null native texture pointer.");
                return CreateBlackFallback(renderGraph, context.targetSize, context.sceneColorFormat);
            }

            var cameraToWorld = BuildCameraToWorld(camera.transform);
            var cameraChanged = !previousCameraValid ||
                                !MatrixNear(previousCameraToWorld, cameraToWorld) ||
                                Mathf.Abs(previousVerticalFov - camera.fieldOfView) > CameraChangeEpsilon;
            var resetAccumulation = resized || cameraChanged;
            previousCameraToWorld = cameraToWorld;
            previousVerticalFov = camera.fieldOfView;
            previousCameraValid = true;

            var output = renderGraph.ImportTexture(
                cameraResources.color,
                new ImportResourceParams
                {
                    clearOnFirstUse = true,
                    clearColor = Color.black,
                    discardOnLastUse = false
                });
            OfflinePathTracePass.Record(
                renderGraph,
                output,
                outputPointer,
                context.targetSize,
                cameraToWorld,
                camera.fieldOfView,
                context.preExposure,
                resetAccumulation);
            return new YutrelRendererOutput(output);
        }

        protected override void Dispose(bool disposing)
        {
            cameraResources.Release();
            cameraLocked = false;
            previousCameraValid = false;
            sceneUploadState = SceneUploadState.NotAttempted;

            if (nativeAcquired)
            {
                NativeBridge.Release();
            }
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
                    $"Yutrel Offline Renderer stage 1A requires exactly one enabled MeshFilter + MeshRenderer; found {meshCount}.");
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
                    $"Yutrel Offline Renderer stage 1A requires exactly one enabled Directional Light; found {lightCount}.");
                return false;
            }

            var mesh = selectedFilter.sharedMesh;
            if (mesh == null || !mesh.isReadable)
            {
                NativeBridge.ReportErrorOnce("The stage 1A Mesh must exist and have Read/Write enabled.");
                return false;
            }
            var vertices = mesh.vertices;
            var normals = mesh.normals;
            if (vertices.Length == 0 || normals.Length != vertices.Length)
            {
                NativeBridge.ReportErrorOnce("The stage 1A Mesh must provide one normal per vertex.");
                return false;
            }

            var indices = new List<uint>();
            for (var subMesh = 0; subMesh < mesh.subMeshCount; subMesh++)
            {
                if (mesh.GetTopology(subMesh) != MeshTopology.Triangles)
                {
                    NativeBridge.ReportErrorOnce("Every stage 1A Mesh submesh must use Triangle topology.");
                    return false;
                }
                foreach (var index in mesh.GetIndices(subMesh))
                {
                    indices.Add((uint)index);
                }
            }
            if (indices.Count == 0 || indices.Count % 3 != 0)
            {
                NativeBridge.ReportErrorOnce("The stage 1A Mesh has no valid triangle indices.");
                return false;
            }

            var lightDirection = -selectedLight.transform.forward;
            if (lightDirection.sqrMagnitude < 1e-12f)
            {
                NativeBridge.ReportErrorOnce("The stage 1A Directional Light direction is invalid.");
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

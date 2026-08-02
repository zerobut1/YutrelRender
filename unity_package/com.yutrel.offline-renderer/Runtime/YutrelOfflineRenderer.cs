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

            internal void Release()
            {
                color?.Release();
                color = null;
                width = 0;
                height = 0;
            }
        }

        private readonly Dictionary<int, CameraResources> cameraResources = new();
        private readonly bool nativeAcquired;

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

            var resources = GetOrCreateResources(context.camera, context.targetSize);
            var renderTexture = resources.color?.rt;
            if (renderTexture == null || !renderTexture.IsCreated())
            {
                NativeBridge.ReportErrorOnce("Yutrel Offline Renderer failed to create its camera RenderTexture.");
                return CreateBlackFallback(renderGraph, context.targetSize, context.sceneColorFormat);
            }

            var outputPointer = renderTexture.GetNativeTexturePtr();
            if (outputPointer == IntPtr.Zero)
            {
                NativeBridge.ReportErrorOnce("Yutrel Offline Renderer received a null native texture pointer.");
                return CreateBlackFallback(renderGraph, context.targetSize, context.sceneColorFormat);
            }

            var output = renderGraph.ImportTexture(
                resources.color,
                new ImportResourceParams
                {
                    clearOnFirstUse = true,
                    clearColor = Color.black,
                    discardOnLastUse = false
                });
            OfflineClearPass.Record(renderGraph, output, outputPointer);
            return new YutrelRendererOutput(output);
        }

        protected override void Dispose(bool disposing)
        {
            foreach (var resources in cameraResources.Values)
            {
                resources.Release();
            }
            cameraResources.Clear();

            if (nativeAcquired)
            {
                NativeBridge.Release();
            }
        }

        private CameraResources GetOrCreateResources(Camera camera, Vector2Int targetSize)
        {
            var cameraId = camera.GetEntityId().GetHashCode();
            if (!cameraResources.TryGetValue(cameraId, out var resources))
            {
                resources = new CameraResources();
                cameraResources.Add(cameraId, resources);
            }

            if (resources.color != null &&
                resources.width == targetSize.x &&
                resources.height == targetSize.y)
            {
                return resources;
            }

            resources.Release();
            resources.width = targetSize.x;
            resources.height = targetSize.y;
            resources.color = RTHandles.Alloc(
                targetSize.x,
                targetSize.y,
                colorFormat: GraphicsFormat.R16G16B16A16_SFloat,
                enableRandomWrite: true,
                filterMode: FilterMode.Bilinear,
                wrapMode: TextureWrapMode.Clamp,
                name: $"Yutrel Offline Color ({camera.name})");
            return resources;
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

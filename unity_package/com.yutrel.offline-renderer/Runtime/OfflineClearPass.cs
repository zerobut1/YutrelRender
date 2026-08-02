using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;

namespace Yutrel.OfflineRenderer
{
    internal sealed class OfflinePathTracePass
    {
        private static readonly ProfilingSampler Sampler = new("Yutrel Path Tracing");

        private TextureHandle output;
        private IntPtr outputPointer;
        private Vector2Int size;
        private Matrix4x4 cameraToWorld;
        private float verticalFovDegrees;
        private float preExposure;
        private bool resetAccumulation;

        internal static void Record(
            RenderGraph renderGraph,
            TextureHandle output,
            IntPtr outputPointer,
            Vector2Int size,
            Matrix4x4 cameraToWorld,
            float verticalFovDegrees,
            float preExposure,
            bool resetAccumulation)
        {
            using var builder = renderGraph.AddComputePass<OfflinePathTracePass>(
                Sampler.name,
                out var pass,
                Sampler);

            pass.output = output;
            pass.outputPointer = outputPointer;
            pass.size = size;
            pass.cameraToWorld = cameraToWorld;
            pass.verticalFovDegrees = verticalFovDegrees;
            pass.preExposure = preExposure;
            pass.resetAccumulation = resetAccumulation;

            builder.UseTexture(pass.output, AccessFlags.Write);
            builder.EnableAsyncCompute(false);
            builder.AllowGlobalStateModification(true);
            builder.SetRenderFunc<OfflinePathTracePass>(static (data, context) =>
            {
                NativeBridge.IssuePathTrace(
                    context.cmd,
                    data.outputPointer,
                    data.size,
                    data.cameraToWorld,
                    data.verticalFovDegrees,
                    data.preExposure,
                    data.resetAccumulation);
            });
        }
    }
}

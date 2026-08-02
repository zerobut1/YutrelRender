using System;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;

namespace Yutrel.OfflineRenderer
{
    internal sealed class OfflineClearPass
    {
        private static readonly ProfilingSampler Sampler = new("Luisa Fixed Color");

        private TextureHandle output;
        private IntPtr outputPointer;

        internal static void Record(
            RenderGraph renderGraph,
            TextureHandle output,
            IntPtr outputPointer)
        {
            using var builder = renderGraph.AddComputePass<OfflineClearPass>(
                Sampler.name,
                out var pass,
                Sampler);

            pass.output = output;
            pass.outputPointer = outputPointer;

            builder.UseTexture(pass.output, AccessFlags.Write);
            builder.EnableAsyncCompute(false);
            builder.AllowGlobalStateModification(true);
            builder.SetRenderFunc<OfflineClearPass>(static (data, context) =>
            {
                NativeBridge.IssueClear(context.cmd, data.outputPointer);
            });
        }
    }
}

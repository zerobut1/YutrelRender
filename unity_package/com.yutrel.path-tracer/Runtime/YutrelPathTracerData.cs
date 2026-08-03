using UnityEngine;
using YutrelRP;

namespace Yutrel.PathTracer
{
    [CreateAssetMenu(
        fileName = "YutrelPathTracer",
        menuName = "Rendering/YutrelPathTracer")]
    public sealed class YutrelPathTracerData : YutrelRendererData
    {
        protected override YutrelRenderer CreateRenderer()
        {
            return new YutrelPathTracer();
        }
    }
}

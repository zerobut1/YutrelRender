using UnityEngine;
using YutrelRP;

namespace Yutrel.OfflineRenderer
{
    [CreateAssetMenu(
        fileName = "YutrelOfflineRenderer",
        menuName = "Rendering/Yutrel Offline Renderer")]
    public sealed class YutrelOfflineRendererData : YutrelRendererData
    {
        protected override YutrelRenderer CreateRenderer()
        {
            return new YutrelOfflineRenderer();
        }
    }
}

using UnityEngine;

namespace Yutrel.PathTracer
{
    internal readonly struct OpenPBRMaterialData
    {
        internal readonly float baseWeight;
        internal readonly Vector3 baseColor;
        internal readonly float baseMetalness;
        internal readonly float baseDiffuseRoughness;
        internal readonly float specularWeight;
        internal readonly Vector3 specularColor;
        internal readonly float specularRoughness;
        internal readonly float specularRoughnessAnisotropy;
        internal readonly float specularIor;

        internal OpenPBRMaterialData(
            float baseWeight,
            Vector3 baseColor,
            float baseMetalness,
            float baseDiffuseRoughness,
            float specularWeight,
            Vector3 specularColor,
            float specularRoughness,
            float specularRoughnessAnisotropy,
            float specularIor)
        {
            this.baseWeight = baseWeight;
            this.baseColor = baseColor;
            this.baseMetalness = baseMetalness;
            this.baseDiffuseRoughness = baseDiffuseRoughness;
            this.specularWeight = specularWeight;
            this.specularColor = specularColor;
            this.specularRoughness = specularRoughness;
            this.specularRoughnessAnisotropy = specularRoughnessAnisotropy;
            this.specularIor = specularIor;
        }
    }

    internal static class OpenPBRMaterialAdapter
    {
        internal const string ShaderName = "YutrelRP/OpenPBR";

        private static readonly int BaseWeightId = Shader.PropertyToID("_OpenPBRBaseWeight");
        private static readonly int BaseColorId = Shader.PropertyToID("_OpenPBRBaseColor");
        private static readonly int BaseMetalnessId = Shader.PropertyToID("_OpenPBRBaseMetalness");
        private static readonly int BaseDiffuseRoughnessId =
            Shader.PropertyToID("_OpenPBRBaseDiffuseRoughness");
        private static readonly int SpecularWeightId = Shader.PropertyToID("_OpenPBRSpecularWeight");
        private static readonly int SpecularColorId = Shader.PropertyToID("_OpenPBRSpecularColor");
        private static readonly int SpecularRoughnessId = Shader.PropertyToID("_OpenPBRSpecularRoughness");
        private static readonly int SpecularRoughnessAnisotropyId =
            Shader.PropertyToID("_OpenPBRSpecularRoughnessAnisotropy");
        private static readonly int SpecularIorId = Shader.PropertyToID("_OpenPBRSpecularIOR");

        internal static bool IsOpenPBR(Material material)
        {
            return material != null && material.shader != null &&
                   material.shader.name == ShaderName;
        }

        internal static bool TryRead(Material material, out OpenPBRMaterialData data)
        {
            data = default;
            if (!IsOpenPBR(material) ||
                !material.HasProperty(BaseWeightId) ||
                !material.HasProperty(BaseColorId) ||
                !material.HasProperty(BaseMetalnessId) ||
                !material.HasProperty(BaseDiffuseRoughnessId) ||
                !material.HasProperty(SpecularWeightId) ||
                !material.HasProperty(SpecularColorId) ||
                !material.HasProperty(SpecularRoughnessId) ||
                !material.HasProperty(SpecularRoughnessAnisotropyId) ||
                !material.HasProperty(SpecularIorId))
            {
                return false;
            }

            var baseWeight = material.GetFloat(BaseWeightId);
            var baseColorValue = material.GetColor(BaseColorId).linear;
            var baseColor = new Vector3(
                baseColorValue.r,
                baseColorValue.g,
                baseColorValue.b);
            var baseMetalness = material.GetFloat(BaseMetalnessId);
            var baseDiffuseRoughness = material.GetFloat(BaseDiffuseRoughnessId);
            var specularWeight = material.GetFloat(SpecularWeightId);
            var specularColorValue = material.GetColor(SpecularColorId).linear;
            var specularColor = new Vector3(
                specularColorValue.r,
                specularColorValue.g,
                specularColorValue.b);
            var specularRoughness = material.GetFloat(SpecularRoughnessId);
            var specularRoughnessAnisotropy = material.GetFloat(SpecularRoughnessAnisotropyId);
            var specularIor = material.GetFloat(SpecularIorId);
            if (!IsFinite(baseWeight) || !IsFinite(baseColor) ||
                !IsFinite(baseMetalness) || !IsFinite(baseDiffuseRoughness) ||
                !IsFinite(specularWeight) || !IsFinite(specularColor) ||
                !IsFinite(specularRoughness) ||
                !IsFinite(specularRoughnessAnisotropy) || !IsFinite(specularIor))
            {
                return false;
            }

            data = new OpenPBRMaterialData(
                Mathf.Clamp01(baseWeight),
                Vector3.Max(baseColor, Vector3.zero),
                Mathf.Clamp01(baseMetalness),
                Mathf.Clamp01(baseDiffuseRoughness),
                Mathf.Max(specularWeight, 0.0f),
                Vector3.Max(specularColor, Vector3.zero),
                Mathf.Clamp01(specularRoughness),
                Mathf.Clamp01(specularRoughnessAnisotropy),
                Mathf.Max(specularIor, 1.0e-6f));
            return true;
        }

        internal static int ComputeSignature(Material material)
        {
            if (!IsOpenPBR(material))
            {
                return 0;
            }
            unchecked
            {
                var hash = material.GetFloat(BaseWeightId).GetHashCode();
                hash = hash * 397 ^ material.GetColor(BaseColorId).GetHashCode();
                hash = hash * 397 ^ material.GetFloat(BaseMetalnessId).GetHashCode();
                hash = hash * 397 ^ material.GetFloat(BaseDiffuseRoughnessId).GetHashCode();
                hash = hash * 397 ^ material.GetFloat(SpecularWeightId).GetHashCode();
                hash = hash * 397 ^ material.GetColor(SpecularColorId).GetHashCode();
                hash = hash * 397 ^ material.GetFloat(SpecularRoughnessId).GetHashCode();
                hash = hash * 397 ^ material.GetFloat(SpecularRoughnessAnisotropyId).GetHashCode();
                hash = hash * 397 ^ material.GetFloat(SpecularIorId).GetHashCode();
                return hash;
            }
        }

        private static bool IsFinite(Vector3 value)
        {
            return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }
    }
}

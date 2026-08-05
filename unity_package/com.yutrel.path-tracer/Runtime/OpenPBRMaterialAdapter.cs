using System;
using System.Collections.Generic;
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

    internal enum OpenPBRParameter
    {
        BaseWeight,
        BaseColor,
        BaseMetalness,
        BaseDiffuseRoughness,
        SpecularWeight,
        SpecularColor,
        SpecularRoughness,
        SpecularRoughnessAnisotropy,
        SpecularIOR,
    }

    /// <summary>Target texture slot in the native OpenPBR surface ABI.</summary>
    internal enum OpenPBRTextureSlot
    {
        BaseColor,
        Normal,
        SpecularRoughness,
        BaseMetalness,
        MaterialAo,
    }

    /// <summary>
    /// Maps one material texture property onto one OpenPBR texture slot.
    /// `isSRGB` declares the texture encoding (sRGB maps are decoded to linear
    /// on the native side); `invert` applies 1-x per pixel on the host and is
    /// used for the Sponza smoothness -> roughness conversion.
    /// </summary>
    internal readonly struct OpenPBRTextureMapping
    {
        internal readonly string propertyName;
        internal readonly OpenPBRTextureSlot slot;
        internal readonly bool isSRGB;
        internal readonly bool invert;

        internal OpenPBRTextureMapping(string propertyName, OpenPBRTextureSlot slot,
            bool isSRGB = false, bool invert = false)
        {
            this.propertyName = propertyName;
            this.slot = slot;
            this.isSRGB = isSRGB;
            this.invert = invert;
        }
    }

    /// <summary>
    /// Maps one material property onto one OpenPBR parameter.
    /// When <see cref="propertyName"/> is null the mapping is a constant:
    /// <see cref="constantValue"/> is used directly (e.g. Sponza's hard-coded
    /// specular_weight = 0.5, which has no material property backing it).
    /// </summary>
    internal readonly struct OpenPBRPropertyMapping
    {
        internal readonly string propertyName; // null = constant mapping
        internal readonly OpenPBRParameter parameter;
        internal readonly bool invert;  // value -> 1 - value (e.g. smoothness -> roughness)
        internal readonly bool isColor; // read via GetColor(...).linear instead of GetFloat
        internal readonly float constantValue; // used when propertyName == null

        internal OpenPBRPropertyMapping(string propertyName, OpenPBRParameter parameter,
            bool invert = false, bool isColor = false, float constantValue = 0.0f)
        {
            this.propertyName = propertyName;
            this.parameter = parameter;
            this.invert = invert;
            this.isColor = isColor;
            this.constantValue = constantValue;
        }
    }

    /// <summary>
    /// Bridges Unity materials to the YutrelRender OpenPBR surface parameters.
    ///
    /// Recognition is generic: a material is OpenPBR iff its shader carries the
    /// SubShader tag "YutrelMaterialType" = "OpenPBR". Parameter extraction uses
    /// a per-shader property mapping table:
    ///   - Shaders without a registered mapping use the standard OpenPBR
    ///     property names (_OpenPBRBaseWeight, ...), which must all be present.
    ///   - Custom shaders (e.g. YutrelRP/Sponza/OpenPBR) register their own
    ///     property mapping via <see cref="RegisterShaderMapping"/>, so the
    ///     plugin needs no code changes to support new OpenPBR shaders.
    /// </summary>
    internal static class OpenPBRMaterialAdapter
    {
        internal const string MaterialTypeTag = "YutrelMaterialType";
        internal const string MaterialTypeValue = "OpenPBR";

        private static readonly OpenPBRPropertyMapping[] DefaultMappings =
        {
            new OpenPBRPropertyMapping("_OpenPBRBaseWeight", OpenPBRParameter.BaseWeight),
            new OpenPBRPropertyMapping("_OpenPBRBaseColor", OpenPBRParameter.BaseColor, isColor: true),
            new OpenPBRPropertyMapping("_OpenPBRBaseMetalness", OpenPBRParameter.BaseMetalness),
            new OpenPBRPropertyMapping("_OpenPBRBaseDiffuseRoughness", OpenPBRParameter.BaseDiffuseRoughness),
            new OpenPBRPropertyMapping("_OpenPBRSpecularWeight", OpenPBRParameter.SpecularWeight),
            new OpenPBRPropertyMapping("_OpenPBRSpecularColor", OpenPBRParameter.SpecularColor, isColor: true),
            new OpenPBRPropertyMapping("_OpenPBRSpecularRoughness", OpenPBRParameter.SpecularRoughness),
            new OpenPBRPropertyMapping("_OpenPBRSpecularRoughnessAnisotropy", OpenPBRParameter.SpecularRoughnessAnisotropy),
            new OpenPBRPropertyMapping("_OpenPBRSpecularIOR", OpenPBRParameter.SpecularIOR),
        };

        private static readonly Dictionary<string, OpenPBRPropertyMapping[]> ShaderMappings = new();
        private static readonly Dictionary<string, OpenPBRTextureMapping[]> ShaderTextureMappings = new();
        private static readonly Dictionary<string, int> ShaderCullModeOffValues = new();

        static OpenPBRMaterialAdapter()
        {
            OpenPBRShaderMappings.RegisterAll(RegisterShaderMapping, RegisterTextureMapping, RegisterCullModeOffValue);
        }

        /// <summary>Registers a property mapping for a shader (by shader name).</summary>
        internal static void RegisterShaderMapping(string shaderName, OpenPBRPropertyMapping[] mappings)
        {
            ShaderMappings[shaderName] = mappings;
        }

        /// <summary>Registers a texture mapping for a shader (by shader name).</summary>
        internal static void RegisterTextureMapping(string shaderName, OpenPBRTextureMapping[] mappings)
        {
            ShaderTextureMappings[shaderName] = mappings;
        }

        /// <summary>
        /// Registers the numeric value of this shader's "Cull Off" state.
        /// Standard shaders use CullMode.Off == 0; Sponza's
        /// [Enum(Off,2,On,0)] uses 2. Defaults to CullMode.Off (0) when unset.
        /// </summary>
        internal static void RegisterCullModeOffValue(string shaderName, int value)
        {
            ShaderCullModeOffValues[shaderName] = value;
        }

        /// <summary>
        /// The numeric value that means "double-sided" for this material's
        /// shader (i.e. its Cull Off state), or CullMode.Off (0) when the
        /// shader registered nothing.
        /// </summary>
        internal static int CullModeOffValueFor(Material material)
        {
            if (material != null && material.shader != null &&
                ShaderCullModeOffValues.TryGetValue(material.shader.name, out var value))
            {
                return value;
            }
            return (int)UnityEngine.Rendering.CullMode.Off;
        }

        internal static bool IsOpenPBR(Material material)
        {
            return material != null && material.shader != null &&
                   material.GetTag(MaterialTypeTag, true, string.Empty) == MaterialTypeValue;
        }

        /// <summary>
        /// Extracts the 9 OpenPBR base-layer scalars. Returns false when the
        /// material is not OpenPBR or when a standard OpenPBR material is
        /// missing any of the required properties.
        /// </summary>
        internal static bool TryRead(Material material, out OpenPBRMaterialData data)
        {
            data = default;
            if (!IsOpenPBR(material))
            {
                return false;
            }

            var mappings = GetMappings(material);
            var hasRegisteredMapping = HasRegisteredMapping(material);
            var baseWeight = 1.0f;
            var baseColor = new Vector3(0.8f, 0.8f, 0.8f);
            var baseMetalness = 0.0f;
            var baseDiffuseRoughness = 0.0f;
            var specularWeight = 1.0f;
            var specularColor = Vector3.one;
            var specularRoughness = 0.3f;
            var specularRoughnessAnisotropy = 0.0f;
            var specularIor = 1.5f;
            var foundAny = false;

            foreach (var mapping in mappings)
            {
                float value;
                if (mapping.propertyName == null)
                {
                    // Constant mapping: no material property behind it.
                    value = mapping.constantValue;
                }
                else
                {
                    if (!material.HasProperty(mapping.propertyName))
                    {
                        // Standard OpenPBR materials must expose every parameter;
                        // custom mappings may omit parameters (defaults are used).
                        if (!hasRegisteredMapping)
                        {
                            return false;
                        }
                        continue;
                    }

                    if (mapping.isColor)
                    {
                        var colorValue = material.GetColor(mapping.propertyName).linear;
                        var color = new Vector3(colorValue.r, colorValue.g, colorValue.b);
                        switch (mapping.parameter)
                        {
                            case OpenPBRParameter.BaseColor: baseColor = color; break;
                            case OpenPBRParameter.SpecularColor: specularColor = color; break;
                        }
                        foundAny = true;
                        continue;
                    }

                    value = material.GetFloat(mapping.propertyName);
                    if (mapping.invert)
                    {
                        value = 1.0f - value;
                    }
                }
                foundAny = true;
                switch (mapping.parameter)
                {
                    case OpenPBRParameter.BaseWeight: baseWeight = value; break;
                    case OpenPBRParameter.BaseMetalness: baseMetalness = value; break;
                    case OpenPBRParameter.BaseDiffuseRoughness: baseDiffuseRoughness = value; break;
                    case OpenPBRParameter.SpecularWeight: specularWeight = value; break;
                    case OpenPBRParameter.SpecularRoughness: specularRoughness = value; break;
                    case OpenPBRParameter.SpecularRoughnessAnisotropy: specularRoughnessAnisotropy = value; break;
                    case OpenPBRParameter.SpecularIOR: specularIor = value; break;
                }
            }

            if (!foundAny ||
                !IsFinite(baseWeight) || !IsFinite(baseColor) ||
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

        /// <summary>Content hash of the mapped properties (for material caching).</summary>
        internal static int ComputeSignature(Material material)
        {
            if (!IsOpenPBR(material))
            {
                return 0;
            }
            unchecked
            {
                var hash = 17;
                foreach (var mapping in GetMappings(material))
                {
                    if (mapping.propertyName == null ||
                        !material.HasProperty(mapping.propertyName))
                    {
                        continue;
                    }
                    hash = hash * 397 ^ (mapping.isColor
                        ? material.GetColor(mapping.propertyName).GetHashCode()
                        : material.GetFloat(mapping.propertyName).GetHashCode());
                }
                return hash;
            }
        }

        private static OpenPBRPropertyMapping[] GetMappings(Material material)
        {
            if (material.shader != null &&
                ShaderMappings.TryGetValue(material.shader.name, out var mappings))
            {
                return mappings;
            }
            return DefaultMappings;
        }

        /// <summary>
        /// Texture mappings for the material's shader, or an empty array when
        /// the shader registered none (standard OpenPBR materials have no
        /// textures).
        /// </summary>
        internal static OpenPBRTextureMapping[] GetTextureMappings(Material material)
        {
            if (material.shader != null &&
                ShaderTextureMappings.TryGetValue(material.shader.name, out var mappings))
            {
                return mappings;
            }
            return Array.Empty<OpenPBRTextureMapping>();
        }

        private static bool HasRegisteredMapping(Material material)
        {
            return material.shader != null && ShaderMappings.ContainsKey(material.shader.name);
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

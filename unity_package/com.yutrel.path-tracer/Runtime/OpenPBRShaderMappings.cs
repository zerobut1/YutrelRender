namespace Yutrel.PathTracer
{
    /// <summary>
    /// Central registry of per-shader OpenPBR property mappings.
    ///
    /// Every shader that carries the "YutrelMaterialType" = "OpenPBR" tag is
    /// recognized by the path tracer; shaders that do not use the standard
    /// _OpenPBR* property names register their own mapping here. To support a
    /// new OpenPBR shader, add one RegisterShaderMapping/RegisterTextureMapping
    /// call in RegisterAll — no other plugin code changes are needed.
    /// </summary>
    internal static class OpenPBRShaderMappings
    {
        internal static void RegisterAll(
            System.Action<string, OpenPBRPropertyMapping[]> register,
            System.Action<string, OpenPBRTextureMapping[]> registerTextures,
            System.Action<string, int> registerCullModeOffValue)
        {
            // YutrelRP/Sponza/OpenPBR: uses Sponza property names; the smoothness
            // map is inverted to roughness. The shader hard-codes specular_weight
            // = 0.5 (see Sponza_OpenPBRSurface.hlsl) with no backing property, so
            // it is registered as a constant mapping. Cull mode uses the Sponza
            // enum [Enum(Off,2,On,0)], so "Cull Off" (double-sided) is 2, not 0.
            register("YutrelRP/Sponza/OpenPBR", new[]
            {
                new OpenPBRPropertyMapping("_BaseColor", OpenPBRParameter.BaseColor, isColor: true),
                new OpenPBRPropertyMapping("_Smoothness", OpenPBRParameter.SpecularRoughness, invert: true),
                new OpenPBRPropertyMapping(null, OpenPBRParameter.SpecularWeight, constantValue: 0.5f),
            });
            registerCullModeOffValue("YutrelRP/Sponza/OpenPBR", 2);

            // Sponza textures. _MaterialAOTex is uploaded for completeness but
            // is not applied by the renderer (OpenPBRSurface has no AO
            // parameter, matching the YutrelRP deferred path).
            registerTextures("YutrelRP/Sponza/OpenPBR", new[]
            {
                new OpenPBRTextureMapping("_BaseColorTex", OpenPBRTextureSlot.BaseColor, isSRGB: true),
                new OpenPBRTextureMapping("_NormalTex", OpenPBRTextureSlot.Normal),
                new OpenPBRTextureMapping("_SmoothnessTex", OpenPBRTextureSlot.SpecularRoughness, invert: true),
                new OpenPBRTextureMapping("_MetallicTex", OpenPBRTextureSlot.BaseMetalness),
                new OpenPBRTextureMapping("_MaterialAOTex", OpenPBRTextureSlot.MaterialAo),
            });
        }
    }
}

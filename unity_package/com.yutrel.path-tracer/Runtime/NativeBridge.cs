using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;

namespace Yutrel.PathTracer
{
    internal static class NativeBridge
    {
        private const string LibraryName = "YutrelUnityPlugin";
        private const uint AbiVersion = 10;
        private const int MaxEnvironmentPathByteCount = 32768;
        private const int ClearEventId = 0;
        private const int PathTraceEventId = 1;

        private static readonly object Sync = new();
        private static readonly HashSet<string> ReportedMessages = new();
        private static int referenceCount;
        private static IntPtr renderEvent;

        [StructLayout(LayoutKind.Sequential)]
        private struct ClearEventData
        {
            public uint abiVersion;
            public uint structSize;
            public IntPtr output;
        }

        internal enum SceneMeshOperation : uint
        {
            AddOrReplace = 1,
            Transform = 2,
            Remove = 3
        }

        internal enum ExternalTextureEncoding : uint
        {
            LinearSrgb = 0,
            Srgb = 1
        }

        internal enum SceneMaterialType : uint
        {
            FallbackDiffuse = 0,
            OpenPBR = 1
        }

        [StructLayout(LayoutKind.Sequential)]
        private unsafe struct OpenPBRMaterialAbi
        {
            public float baseWeight;
            public fixed float baseColor[3];
            public float baseMetalness;
            public float baseDiffuseRoughness;
            public float specularWeight;
            public fixed float specularColor[3];
            public float specularRoughness;
            public float specularRoughnessAnisotropy;
            public float specularIor;
        }

        [StructLayout(LayoutKind.Sequential)]
        private unsafe struct OpenPBRTextureSlotAbi
        {
            public IntPtr pixels; // null = unused slot (parameter falls back to constant)
            public uint width;
            public uint height;
            public ExternalTextureEncoding encoding;
            public fixed float uvScale[2];
            public fixed float uvOffset[2];
        }

        [StructLayout(LayoutKind.Sequential)]
        private unsafe struct SceneSubMeshData
        {
            public uint indexOffset;
            public uint indexCount;
            public fixed float emissiveColor[3];
            public float emissiveLuminanceNits;
            public uint doubleSided;
            public ExternalTextureEncoding textureEncoding;
            public IntPtr emissivePixels;
            public uint textureWidth;
            public uint textureHeight;
            public fixed float uvScale[2];
            public fixed float uvOffset[2];
            public ulong materialId;
            public SceneMaterialType materialType;
            public OpenPBRMaterialAbi openPbr;
            public OpenPBRTextureSlotAbi baseColor;
            public OpenPBRTextureSlotAbi normal;
            public OpenPBRTextureSlotAbi specularRoughness;
            public OpenPBRTextureSlotAbi baseMetalness;
            public OpenPBRTextureSlotAbi materialAo;
        }

        [StructLayout(LayoutKind.Sequential)]
        private unsafe struct SceneMeshDelta
        {
            public ulong meshId;
            public SceneMeshOperation operation;
            public uint reserved;
            public IntPtr positions;
            public IntPtr normals;
            public IntPtr uvs;
            public IntPtr indices;
            public IntPtr subMeshes;
            public uint vertexCount;
            public uint indexCount;
            public uint subMeshCount;
            public uint subMeshStructSize;
            public fixed float localToWorld[16];
        }

        [StructLayout(LayoutKind.Sequential)]
        private unsafe struct DirectionalLightData
        {
            public fixed float color[3];
            public float illuminanceLux;
            public fixed float direction[3];
            public uint enabled;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct EnvironmentData
        {
            public IntPtr pathUtf8;
            public uint pathByteCount;
            public float intensity;
            public uint enabled;
        }

        [StructLayout(LayoutKind.Sequential)]
        private unsafe struct SceneMaterialUpdateData
        {
            public ulong meshId;
            public uint subMeshIndex;
            public ulong materialId;
            public uint doubleSided;
            public OpenPBRMaterialAbi openPbr;
        }

        [StructLayout(LayoutKind.Sequential)]
        private unsafe struct SceneDeltaData
        {
            public uint abiVersion;
            public uint structSize;
            public ulong revision;
            public IntPtr meshes;
            public uint meshCount;
            public uint meshStructSize;
            public DirectionalLightData light;
            public uint lightChanged;
            public uint environmentChanged;
            public IntPtr materialUpdates;
            public uint materialUpdateCount;
            public uint materialUpdateStructSize;
            public EnvironmentData environment;
        }

        internal readonly struct SceneMeshUpdate
        {
            internal readonly ulong id;
            internal readonly SceneMeshOperation operation;
            internal readonly Vector3[] positions;
            internal readonly Vector3[] normals;
            internal readonly Vector2[] uvs;
            internal readonly uint[] indices;
            internal readonly SceneSubMeshUpdate[] subMeshes;
            internal readonly Matrix4x4 localToWorld;

            private SceneMeshUpdate(
                ulong id,
                SceneMeshOperation operation,
                Vector3[] positions,
                Vector3[] normals,
                Vector2[] uvs,
                uint[] indices,
                SceneSubMeshUpdate[] subMeshes,
                Matrix4x4 localToWorld)
            {
                this.id = id;
                this.operation = operation;
                this.positions = positions;
                this.normals = normals;
                this.uvs = uvs;
                this.indices = indices;
                this.subMeshes = subMeshes;
                this.localToWorld = localToWorld;
            }

            internal static SceneMeshUpdate AddOrReplace(
                ulong id,
                Vector3[] positions,
                Vector3[] normals,
                Vector2[] uvs,
                uint[] indices,
                SceneSubMeshUpdate[] subMeshes,
                Matrix4x4 localToWorld)
            {
                return new SceneMeshUpdate(
                    id,
                    SceneMeshOperation.AddOrReplace,
                    positions,
                    normals,
                    uvs,
                    indices,
                    subMeshes,
                    localToWorld);
            }

            internal static SceneMeshUpdate Transform(ulong id, Matrix4x4 localToWorld)
            {
                return new SceneMeshUpdate(
                    id,
                    SceneMeshOperation.Transform,
                    null,
                    null,
                    null,
                    null,
                    null,
                    localToWorld);
            }

            internal static SceneMeshUpdate Remove(ulong id)
            {
                return new SceneMeshUpdate(
                    id,
                    SceneMeshOperation.Remove,
                    null,
                    null,
                    null,
                    null,
                    null,
                    Matrix4x4.identity);
            }
        }

        internal readonly struct OpenPBRTextureSlot
        {
            internal readonly ExternalTextureEncoding encoding;
            internal readonly Color[] pixels; // null = slot unused
            internal readonly uint width;
            internal readonly uint height;
            internal readonly Vector2 uvScale;
            internal readonly Vector2 uvOffset;

            internal OpenPBRTextureSlot(
                ExternalTextureEncoding encoding,
                Color[] pixels,
                uint width,
                uint height,
                Vector2 uvScale,
                Vector2 uvOffset)
            {
                this.encoding = encoding;
                this.pixels = pixels;
                this.width = width;
                this.height = height;
                this.uvScale = uvScale;
                this.uvOffset = uvOffset;
            }

            internal bool HasTexture => pixels != null && width > 0 && height > 0;
        }

        internal readonly struct SceneSubMeshUpdate
        {
            internal readonly uint indexOffset;
            internal readonly uint indexCount;
            internal readonly Vector3 emissiveColorLinearSrgb;
            internal readonly float emissiveLuminanceNits;
            internal readonly bool doubleSided;
            internal readonly ExternalTextureEncoding textureEncoding;
            internal readonly Color[] emissivePixels;
            internal readonly uint textureWidth;
            internal readonly uint textureHeight;
            internal readonly Vector2 uvScale;
            internal readonly Vector2 uvOffset;
            internal readonly ulong materialId;
            internal readonly SceneMaterialType materialType;
            internal readonly OpenPBRMaterialData openPbr;
            internal readonly OpenPBRTextureSlot baseColor;
            internal readonly OpenPBRTextureSlot normal;
            internal readonly OpenPBRTextureSlot specularRoughness;
            internal readonly OpenPBRTextureSlot baseMetalness;
            internal readonly OpenPBRTextureSlot materialAo;

            internal SceneSubMeshUpdate(
                uint indexOffset,
                uint indexCount,
                Vector3 emissiveColorLinearSrgb,
                float emissiveLuminanceNits,
                bool doubleSided,
                ExternalTextureEncoding textureEncoding,
                Color[] emissivePixels,
                uint textureWidth,
                uint textureHeight,
                Vector2 uvScale,
                Vector2 uvOffset,
                ulong materialId,
                SceneMaterialType materialType,
                OpenPBRMaterialData openPbr,
                OpenPBRTextureSlot baseColor,
                OpenPBRTextureSlot normal,
                OpenPBRTextureSlot specularRoughness,
                OpenPBRTextureSlot baseMetalness,
                OpenPBRTextureSlot materialAo)
            {
                this.indexOffset = indexOffset;
                this.indexCount = indexCount;
                this.emissiveColorLinearSrgb = emissiveColorLinearSrgb;
                this.emissiveLuminanceNits = emissiveLuminanceNits;
                this.doubleSided = doubleSided;
                this.textureEncoding = textureEncoding;
                this.emissivePixels = emissivePixels;
                this.textureWidth = textureWidth;
                this.textureHeight = textureHeight;
                this.uvScale = uvScale;
                this.uvOffset = uvOffset;
                this.materialId = materialId;
                this.materialType = materialType;
                this.openPbr = openPbr;
                this.baseColor = baseColor;
                this.normal = normal;
                this.specularRoughness = specularRoughness;
                this.baseMetalness = baseMetalness;
                this.materialAo = materialAo;
            }
        }

        internal readonly struct SceneMaterialUpdate
        {
            internal readonly ulong meshId;
            internal readonly uint subMeshIndex;
            internal readonly ulong materialId;
            internal readonly bool doubleSided;
            internal readonly OpenPBRMaterialData openPbr;

            internal SceneMaterialUpdate(
                ulong meshId,
                uint subMeshIndex,
                ulong materialId,
                bool doubleSided,
                OpenPBRMaterialData openPbr)
            {
                this.meshId = meshId;
                this.subMeshIndex = subMeshIndex;
                this.materialId = materialId;
                this.doubleSided = doubleSided;
                this.openPbr = openPbr;
            }
        }

        internal readonly struct DirectionalLightUpdate
        {
            internal readonly Vector3 colorLinearSrgb;
            internal readonly float illuminanceLux;
            internal readonly Vector3 direction;
            internal readonly bool enabled;

            internal DirectionalLightUpdate(
                Vector3 colorLinearSrgb,
                float illuminanceLux,
                Vector3 direction,
                bool enabled)
            {
                this.colorLinearSrgb = colorLinearSrgb;
                this.illuminanceLux = illuminanceLux;
                this.direction = direction;
                this.enabled = enabled;
            }
        }

        internal readonly struct EnvironmentUpdate
        {
            internal readonly string hdrPath;
            internal readonly float intensity;
            internal readonly bool enabled;

            internal EnvironmentUpdate(string hdrPath, float intensity, bool enabled)
            {
                this.hdrPath = enabled ? hdrPath : null;
                this.intensity = enabled ? intensity : 0.0f;
                this.enabled = enabled;
            }

            internal static EnvironmentUpdate Disabled => new(null, 0.0f, false);
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct PathTraceEventData
        {
            public uint abiVersion;
            public uint structSize;
            public IntPtr output;
            public uint width;
            public uint height;
            public uint viewId;
            public uint flipOutputY;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
            public float[] cameraToWorld;

            public float verticalFovDegrees;
            public float preExposure;
            public uint resetAccumulation;
        }

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        [DllImport(LibraryName, CallingConvention = CallingConvention.Winapi)]
        private static extern int YutrelUnityInitialize();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Winapi)]
        private static extern void YutrelUnityShutdown();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Winapi)]
        private static extern IntPtr YutrelUnityGetRenderEventFunc();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Winapi)]
        private static extern int YutrelUnitySubmitSceneDelta(ref SceneDeltaData data);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Winapi)]
        private static extern IntPtr YutrelUnityCreateClearEvent(ref ClearEventData data);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Winapi)]
        private static extern IntPtr YutrelUnityCreatePathTraceEvent(ref PathTraceEventData data);
#endif

        internal static bool Acquire()
        {
            lock (Sync)
            {
                if (referenceCount > 0)
                {
                    referenceCount++;
                    return true;
                }

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
                if (SystemInfo.graphicsDeviceType != GraphicsDeviceType.Direct3D12)
                {
                    ReportErrorOnce("YutrelPathTracer requires the Direct3D 12 graphics API.");
                    return false;
                }

                try
                {
                    var result = YutrelUnityInitialize();
                    renderEvent = result == 0 ? YutrelUnityGetRenderEventFunc() : IntPtr.Zero;
                    if (result != 0 || renderEvent == IntPtr.Zero)
                    {
                        ReportErrorOnce($"YutrelUnityPlugin initialization failed with code {result}.");
                        return false;
                    }

                    referenceCount = 1;
                    return true;
                }
                catch (Exception exception) when (
                    exception is DllNotFoundException or
                    EntryPointNotFoundException or
                    BadImageFormatException)
                {
                    ReportErrorOnce($"YutrelUnityPlugin could not be loaded: {exception.Message}");
                    return false;
                }
#else
                ReportErrorOnce("YutrelPathTracer currently supports Windows x64 only.");
                return false;
#endif
            }
        }

        internal static void Release()
        {
            lock (Sync)
            {
                if (referenceCount <= 0)
                {
                    return;
                }

                referenceCount--;
                if (referenceCount != 0)
                {
                    return;
                }

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
                YutrelUnityShutdown();
#endif
                renderEvent = IntPtr.Zero;
            }
        }

        internal static unsafe bool SubmitSceneDelta(
            IReadOnlyList<SceneMeshUpdate> meshes,
            IReadOnlyList<SceneMaterialUpdate> materialUpdates,
            ulong revision,
            DirectionalLightUpdate? lightUpdate,
            EnvironmentUpdate? environmentUpdate)
        {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            if (sizeof(OpenPBRMaterialAbi) != 52 || sizeof(OpenPBRTextureSlotAbi) != 40 ||
                sizeof(SceneSubMeshData) != 328 ||
                sizeof(SceneMaterialUpdateData) != 80 || sizeof(EnvironmentData) != 24 ||
                sizeof(SceneDeltaData) != 112)
            {
                ReportErrorOnce("Yutrel scene ABI layout is invalid.");
                return false;
            }
            if (renderEvent == IntPtr.Zero || meshes == null || revision == 0)
            {
                ReportErrorOnce("Yutrel scene delta has invalid data.");
                return false;
            }

            var descriptors = new SceneMeshDelta[meshes.Count];
            var pinnedArrays = new List<GCHandle>(meshes.Count * 6);
            try
            {
                for (var meshIndex = 0; meshIndex < meshes.Count; meshIndex++)
                {
                    var mesh = meshes[meshIndex];
                    if (mesh.id == 0)
                    {
                        ReportErrorOnce("Yutrel scene delta contains an invalid Mesh ID.");
                        return false;
                    }
                    var descriptor = new SceneMeshDelta
                    {
                        meshId = mesh.id,
                        operation = mesh.operation
                    };
                    if (mesh.operation == SceneMeshOperation.AddOrReplace)
                    {
                        if (mesh.positions == null || mesh.normals == null || mesh.indices == null ||
                            mesh.subMeshes == null ||
                            mesh.positions.Length == 0 || mesh.normals.Length != mesh.positions.Length ||
                            (mesh.uvs != null && mesh.uvs.Length != mesh.positions.Length) ||
                            mesh.indices.Length == 0 || mesh.subMeshes.Length == 0)
                        {
                            ReportErrorOnce("Yutrel scene delta contains invalid Mesh geometry.");
                            return false;
                        }
                        var positionsHandle = GCHandle.Alloc(mesh.positions, GCHandleType.Pinned);
                        pinnedArrays.Add(positionsHandle);
                        descriptor.positions = positionsHandle.AddrOfPinnedObject();
                        var normalsHandle = GCHandle.Alloc(mesh.normals, GCHandleType.Pinned);
                        pinnedArrays.Add(normalsHandle);
                        descriptor.normals = normalsHandle.AddrOfPinnedObject();
                        if (mesh.uvs != null)
                        {
                            var uvsHandle = GCHandle.Alloc(mesh.uvs, GCHandleType.Pinned);
                            pinnedArrays.Add(uvsHandle);
                            descriptor.uvs = uvsHandle.AddrOfPinnedObject();
                        }
                        var indicesHandle = GCHandle.Alloc(mesh.indices, GCHandleType.Pinned);
                        pinnedArrays.Add(indicesHandle);
                        descriptor.indices = indicesHandle.AddrOfPinnedObject();

                        var subMeshDescriptors = new SceneSubMeshData[mesh.subMeshes.Length];
                        for (var subMeshIndex = 0; subMeshIndex < mesh.subMeshes.Length; subMeshIndex++)
                        {
                            var subMesh = mesh.subMeshes[subMeshIndex];
                            var subMeshDescriptor = new SceneSubMeshData
                            {
                                indexOffset = subMesh.indexOffset,
                                indexCount = subMesh.indexCount,
                                emissiveLuminanceNits = subMesh.emissiveLuminanceNits,
                                doubleSided = subMesh.doubleSided ? 1u : 0u,
                                textureEncoding = subMesh.textureEncoding,
                                textureWidth = subMesh.textureWidth,
                                textureHeight = subMesh.textureHeight,
                                materialId = subMesh.materialId,
                                materialType = subMesh.materialType
                            };
                            subMeshDescriptor.emissiveColor[0] = subMesh.emissiveColorLinearSrgb.x;
                            subMeshDescriptor.emissiveColor[1] = subMesh.emissiveColorLinearSrgb.y;
                            subMeshDescriptor.emissiveColor[2] = subMesh.emissiveColorLinearSrgb.z;
                            subMeshDescriptor.uvScale[0] = subMesh.uvScale.x;
                            subMeshDescriptor.uvScale[1] = subMesh.uvScale.y;
                            subMeshDescriptor.uvOffset[0] = subMesh.uvOffset.x;
                            subMeshDescriptor.uvOffset[1] = subMesh.uvOffset.y;
                            subMeshDescriptor.openPbr.baseWeight = subMesh.openPbr.baseWeight;
                            subMeshDescriptor.openPbr.baseColor[0] = subMesh.openPbr.baseColor.x;
                            subMeshDescriptor.openPbr.baseColor[1] = subMesh.openPbr.baseColor.y;
                            subMeshDescriptor.openPbr.baseColor[2] = subMesh.openPbr.baseColor.z;
                            subMeshDescriptor.openPbr.baseMetalness = subMesh.openPbr.baseMetalness;
                            subMeshDescriptor.openPbr.baseDiffuseRoughness =
                                subMesh.openPbr.baseDiffuseRoughness;
                            subMeshDescriptor.openPbr.specularWeight = subMesh.openPbr.specularWeight;
                            subMeshDescriptor.openPbr.specularColor[0] = subMesh.openPbr.specularColor.x;
                            subMeshDescriptor.openPbr.specularColor[1] = subMesh.openPbr.specularColor.y;
                            subMeshDescriptor.openPbr.specularColor[2] = subMesh.openPbr.specularColor.z;
                            subMeshDescriptor.openPbr.specularRoughness =
                                subMesh.openPbr.specularRoughness;
                            subMeshDescriptor.openPbr.specularRoughnessAnisotropy =
                                subMesh.openPbr.specularRoughnessAnisotropy;
                            subMeshDescriptor.openPbr.specularIor = subMesh.openPbr.specularIor;
                            if (subMesh.emissivePixels != null)
                            {
                                var pixelsHandle = GCHandle.Alloc(subMesh.emissivePixels, GCHandleType.Pinned);
                                pinnedArrays.Add(pixelsHandle);
                                subMeshDescriptor.emissivePixels = pixelsHandle.AddrOfPinnedObject();
                            }
                            FillOpenPBRTextureSlot(ref subMeshDescriptor.baseColor, subMesh.baseColor, pinnedArrays);
                            FillOpenPBRTextureSlot(ref subMeshDescriptor.normal, subMesh.normal, pinnedArrays);
                            FillOpenPBRTextureSlot(ref subMeshDescriptor.specularRoughness, subMesh.specularRoughness, pinnedArrays);
                            FillOpenPBRTextureSlot(ref subMeshDescriptor.baseMetalness, subMesh.baseMetalness, pinnedArrays);
                            FillOpenPBRTextureSlot(ref subMeshDescriptor.materialAo, subMesh.materialAo, pinnedArrays);
                            subMeshDescriptors[subMeshIndex] = subMeshDescriptor;
                        }
                        var subMeshesHandle = GCHandle.Alloc(subMeshDescriptors, GCHandleType.Pinned);
                        pinnedArrays.Add(subMeshesHandle);
                        descriptor.subMeshes = subMeshesHandle.AddrOfPinnedObject();
                        descriptor.vertexCount = (uint)mesh.positions.Length;
                        descriptor.indexCount = (uint)mesh.indices.Length;
                        descriptor.subMeshCount = (uint)mesh.subMeshes.Length;
                        descriptor.subMeshStructSize = (uint)sizeof(SceneSubMeshData);
                    }
                    var localToWorld = ToColumnMajor(mesh.localToWorld);
                    for (var i = 0; i < localToWorld.Length; i++)
                    {
                        descriptor.localToWorld[i] = localToWorld[i];
                    }
                    descriptors[meshIndex] = descriptor;
                }

                var materialDescriptors =
                    new SceneMaterialUpdateData[materialUpdates?.Count ?? 0];
                for (var i = 0; i < materialDescriptors.Length; i++)
                {
                    var update = materialUpdates[i];
                    var descriptor = new SceneMaterialUpdateData
                    {
                        meshId = update.meshId,
                        subMeshIndex = update.subMeshIndex,
                        materialId = update.materialId,
                        doubleSided = update.doubleSided ? 1u : 0u
                    };
                    descriptor.openPbr.baseWeight = update.openPbr.baseWeight;
                    descriptor.openPbr.baseColor[0] = update.openPbr.baseColor.x;
                    descriptor.openPbr.baseColor[1] = update.openPbr.baseColor.y;
                    descriptor.openPbr.baseColor[2] = update.openPbr.baseColor.z;
                    descriptor.openPbr.baseMetalness = update.openPbr.baseMetalness;
                    descriptor.openPbr.baseDiffuseRoughness =
                        update.openPbr.baseDiffuseRoughness;
                    descriptor.openPbr.specularWeight = update.openPbr.specularWeight;
                    descriptor.openPbr.specularColor[0] = update.openPbr.specularColor.x;
                    descriptor.openPbr.specularColor[1] = update.openPbr.specularColor.y;
                    descriptor.openPbr.specularColor[2] = update.openPbr.specularColor.z;
                    descriptor.openPbr.specularRoughness = update.openPbr.specularRoughness;
                    descriptor.openPbr.specularRoughnessAnisotropy =
                        update.openPbr.specularRoughnessAnisotropy;
                    descriptor.openPbr.specularIor = update.openPbr.specularIor;
                    materialDescriptors[i] = descriptor;
                }
                GCHandle materialHandle = default;
                if (materialDescriptors.Length > 0)
                {
                    materialHandle = GCHandle.Alloc(materialDescriptors, GCHandleType.Pinned);
                    pinnedArrays.Add(materialHandle);
                }

                var environmentData = default(EnvironmentData);
                if (environmentUpdate.HasValue)
                {
                    var environment = environmentUpdate.Value;
                    if (environment.enabled)
                    {
                        if (string.IsNullOrEmpty(environment.hdrPath))
                        {
                            ReportErrorOnce("Yutrel environment has no HDR path.");
                            return false;
                        }
                        var pathUtf8 = Encoding.UTF8.GetBytes(environment.hdrPath);
                        if (pathUtf8.Length == 0 || pathUtf8.Length > MaxEnvironmentPathByteCount)
                        {
                            ReportErrorOnce("Yutrel environment HDR path is too long.");
                            return false;
                        }
                        var pathHandle = GCHandle.Alloc(pathUtf8, GCHandleType.Pinned);
                        pinnedArrays.Add(pathHandle);
                        environmentData.pathUtf8 = pathHandle.AddrOfPinnedObject();
                        environmentData.pathByteCount = (uint)pathUtf8.Length;
                        environmentData.intensity = environment.intensity;
                        environmentData.enabled = 1u;
                    }
                }

                fixed (SceneMeshDelta* meshPointer = descriptors)
                {
                    var data = new SceneDeltaData
                    {
                        abiVersion = AbiVersion,
                        structSize = (uint)sizeof(SceneDeltaData),
                        revision = revision,
                        meshes = (IntPtr)meshPointer,
                        meshCount = (uint)descriptors.Length,
                        meshStructSize = (uint)sizeof(SceneMeshDelta),
                        materialUpdates = materialDescriptors.Length > 0
                            ? materialHandle.AddrOfPinnedObject()
                            : IntPtr.Zero,
                        materialUpdateCount = (uint)materialDescriptors.Length,
                        materialUpdateStructSize = (uint)sizeof(SceneMaterialUpdateData),
                        lightChanged = lightUpdate.HasValue ? 1u : 0u,
                        environmentChanged = environmentUpdate.HasValue ? 1u : 0u,
                        environment = environmentData
                    };
                    if (lightUpdate.HasValue)
                    {
                        var light = lightUpdate.Value;
                        data.light.color[0] = light.colorLinearSrgb.x;
                        data.light.color[1] = light.colorLinearSrgb.y;
                        data.light.color[2] = light.colorLinearSrgb.z;
                        data.light.illuminanceLux = light.illuminanceLux;
                        data.light.direction[0] = light.direction.x;
                        data.light.direction[1] = light.direction.y;
                        data.light.direction[2] = light.direction.z;
                        data.light.enabled = light.enabled ? 1u : 0u;
                    }

                    var result = YutrelUnitySubmitSceneDelta(ref data);
                    if (result != 0)
                    {
                        ReportErrorOnce($"Yutrel scene delta submission failed with code {result}.");
                        return false;
                    }
                }
            }
            finally
            {
                for (var i = 0; i < pinnedArrays.Count; i++)
                {
                    var handle = pinnedArrays[i];
                    if (handle.IsAllocated)
                    {
                        handle.Free();
                    }
                }
            }
            return true;
#else
            return false;
#endif
        }

        private static unsafe void FillOpenPBRTextureSlot(
            ref OpenPBRTextureSlotAbi slot,
            OpenPBRTextureSlot source,
            List<GCHandle> pinnedArrays)
        {
            slot.encoding = source.encoding;
            slot.width = source.width;
            slot.height = source.height;
            slot.uvScale[0] = source.uvScale.x;
            slot.uvScale[1] = source.uvScale.y;
            slot.uvOffset[0] = source.uvOffset.x;
            slot.uvOffset[1] = source.uvOffset.y;
            slot.pixels = IntPtr.Zero;
            if (source.HasTexture)
            {
                var handle = GCHandle.Alloc(source.pixels, GCHandleType.Pinned);
                pinnedArrays.Add(handle);
                slot.pixels = handle.AddrOfPinnedObject();
            }
        }

        internal static void IssuePathTrace(
            ComputeCommandBuffer commandBuffer,
            IntPtr output,
            Vector2Int size,
            uint viewId,
            bool flipOutputY,
            Matrix4x4 cameraToWorld,
            float verticalFovDegrees,
            float preExposure,
            bool resetAccumulation)
        {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            if (renderEvent == IntPtr.Zero || output == IntPtr.Zero)
            {
                ReportErrorOnce("Yutrel Path Tracing event has no valid native output.");
                return;
            }

            var data = new PathTraceEventData
            {
                abiVersion = AbiVersion,
                structSize = (uint)Marshal.SizeOf<PathTraceEventData>(),
                output = output,
                width = (uint)size.x,
                height = (uint)size.y,
                viewId = viewId,
                flipOutputY = flipOutputY ? 1u : 0u,
                cameraToWorld = ToColumnMajor(cameraToWorld),
                verticalFovDegrees = verticalFovDegrees,
                preExposure = preExposure,
                resetAccumulation = resetAccumulation ? 1u : 0u
            };
            var eventData = YutrelUnityCreatePathTraceEvent(ref data);
            if (eventData == IntPtr.Zero)
            {
                ReportErrorOnce("YutrelUnityPlugin failed to allocate Path Tracing event data.");
                return;
            }

            commandBuffer.IssuePluginEventAndData(renderEvent, PathTraceEventId, eventData);
#endif
        }

        internal static void IssueClear(ComputeCommandBuffer commandBuffer, IntPtr output)
        {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            if (renderEvent == IntPtr.Zero || output == IntPtr.Zero)
            {
                ReportErrorOnce("Yutrel fixed-color render event has no valid native output.");
                return;
            }

            var data = new ClearEventData
            {
                abiVersion = AbiVersion,
                structSize = (uint)Marshal.SizeOf<ClearEventData>(),
                output = output
            };
            var eventData = YutrelUnityCreateClearEvent(ref data);
            if (eventData == IntPtr.Zero)
            {
                ReportErrorOnce("YutrelUnityPlugin failed to allocate fixed-color event data.");
                return;
            }

            commandBuffer.IssuePluginEventAndData(renderEvent, ClearEventId, eventData);
#endif
        }

        internal static void ReportErrorOnce(string message)
        {
            if (MarkReported(message))
            {
                Debug.LogError(message);
            }
        }

        internal static void ReportInfoOnce(string message)
        {
            if (MarkReported(message))
            {
                Debug.Log(message);
            }
        }

        private static bool MarkReported(string message)
        {
            lock (Sync)
            {
                return ReportedMessages.Add(message);
            }
        }

        private static float[] ToColumnMajor(Matrix4x4 matrix)
        {
            var values = new float[16];
            for (var column = 0; column < 4; column++)
            {
                for (var row = 0; row < 4; row++)
                {
                    values[column * 4 + row] = matrix[row, column];
                }
            }
            return values;
        }
    }
}

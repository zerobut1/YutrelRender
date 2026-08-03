using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace Yutrel.OfflineRenderer
{
    internal static class NativeBridge
    {
        private const string LibraryName = "YutrelUnityPlugin";
        private const uint AbiVersion = 5;
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

        [StructLayout(LayoutKind.Sequential)]
        private unsafe struct SceneMeshDelta
        {
            public ulong meshId;
            public SceneMeshOperation operation;
            public uint reserved;
            public IntPtr positions;
            public IntPtr normals;
            public IntPtr indices;
            public uint vertexCount;
            public uint indexCount;
            public fixed float localToWorld[16];
        }

        [StructLayout(LayoutKind.Sequential)]
        private unsafe struct DirectionalLightData
        {
            public fixed float color[3];
            public float intensity;
            public fixed float direction[3];
            public uint enabled;
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
        }

        internal readonly struct SceneMeshUpdate
        {
            internal readonly ulong id;
            internal readonly SceneMeshOperation operation;
            internal readonly Vector3[] positions;
            internal readonly Vector3[] normals;
            internal readonly uint[] indices;
            internal readonly Matrix4x4 localToWorld;

            private SceneMeshUpdate(
                ulong id,
                SceneMeshOperation operation,
                Vector3[] positions,
                Vector3[] normals,
                uint[] indices,
                Matrix4x4 localToWorld)
            {
                this.id = id;
                this.operation = operation;
                this.positions = positions;
                this.normals = normals;
                this.indices = indices;
                this.localToWorld = localToWorld;
            }

            internal static SceneMeshUpdate AddOrReplace(
                ulong id,
                Vector3[] positions,
                Vector3[] normals,
                uint[] indices,
                Matrix4x4 localToWorld)
            {
                return new SceneMeshUpdate(
                    id,
                    SceneMeshOperation.AddOrReplace,
                    positions,
                    normals,
                    indices,
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
                    Matrix4x4.identity);
            }
        }

        internal readonly struct DirectionalLightUpdate
        {
            internal readonly Color color;
            internal readonly float intensity;
            internal readonly Vector3 direction;
            internal readonly bool enabled;

            internal DirectionalLightUpdate(
                Color color,
                float intensity,
                Vector3 direction,
                bool enabled)
            {
                this.color = color;
                this.intensity = intensity;
                this.direction = direction;
                this.enabled = enabled;
            }
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
                    ReportErrorOnce("Yutrel Offline Renderer requires the Direct3D 12 graphics API.");
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
                ReportErrorOnce("Yutrel Offline Renderer currently supports Windows x64 only.");
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
            ulong revision,
            DirectionalLightUpdate? lightUpdate)
        {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            if (renderEvent == IntPtr.Zero || meshes == null || revision == 0)
            {
                ReportErrorOnce("Yutrel scene delta has invalid data.");
                return false;
            }

            var descriptors = new SceneMeshDelta[meshes.Count];
            var pinnedArrays = new GCHandle[meshes.Count * 3];
            var pinnedCount = 0;
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
                            mesh.positions.Length == 0 || mesh.normals.Length != mesh.positions.Length ||
                            mesh.indices.Length == 0)
                        {
                            ReportErrorOnce("Yutrel scene delta contains invalid Mesh geometry.");
                            return false;
                        }
                        pinnedArrays[pinnedCount] = GCHandle.Alloc(mesh.positions, GCHandleType.Pinned);
                        descriptor.positions = pinnedArrays[pinnedCount++].AddrOfPinnedObject();
                        pinnedArrays[pinnedCount] = GCHandle.Alloc(mesh.normals, GCHandleType.Pinned);
                        descriptor.normals = pinnedArrays[pinnedCount++].AddrOfPinnedObject();
                        pinnedArrays[pinnedCount] = GCHandle.Alloc(mesh.indices, GCHandleType.Pinned);
                        descriptor.indices = pinnedArrays[pinnedCount++].AddrOfPinnedObject();
                        descriptor.vertexCount = (uint)mesh.positions.Length;
                        descriptor.indexCount = (uint)mesh.indices.Length;
                    }
                    var localToWorld = ToColumnMajor(mesh.localToWorld);
                    for (var i = 0; i < localToWorld.Length; i++)
                    {
                        descriptor.localToWorld[i] = localToWorld[i];
                    }
                    descriptors[meshIndex] = descriptor;
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
                        lightChanged = lightUpdate.HasValue ? 1u : 0u
                    };
                    if (lightUpdate.HasValue)
                    {
                        var light = lightUpdate.Value;
                        data.light.color[0] = light.color.r;
                        data.light.color[1] = light.color.g;
                        data.light.color[2] = light.color.b;
                        data.light.intensity = light.intensity;
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
                for (var i = 0; i < pinnedCount; i++)
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

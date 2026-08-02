using System;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace Yutrel.OfflineRenderer
{
    internal static class NativeBridge
    {
        private const string LibraryName = "YutrelUnityPlugin";
        private const uint AbiVersion = 1;
        private const int ClearEventId = 0;

        private static readonly object Sync = new();
        private static int referenceCount;
        private static IntPtr renderEvent;
        private static bool errorReported;

        [StructLayout(LayoutKind.Sequential)]
        private struct ClearEventData
        {
            public uint abiVersion;
            public uint structSize;
            public IntPtr output;
        }

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        [DllImport(LibraryName, CallingConvention = CallingConvention.Winapi)]
        private static extern int YutrelUnityInitialize();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Winapi)]
        private static extern void YutrelUnityShutdown();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Winapi)]
        private static extern IntPtr YutrelUnityGetRenderEventFunc();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Winapi)]
        private static extern IntPtr YutrelUnityCreateClearEvent(ref ClearEventData data);
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
            if (errorReported)
            {
                return;
            }

            errorReported = true;
            Debug.LogError(message);
        }
    }
}

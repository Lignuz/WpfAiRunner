using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;

namespace OnnxEngines.Utils;

public static class OnnxHelper
{
    private static void Log(string message)
    {
        // Visual Studio 출력 창에 표시
        Debug.WriteLine(message);
        // 콘솔에도 표시 (dotnet run 시)
        Console.WriteLine(message);
    }

    public static (InferenceSession Session, string DeviceMode) LoadSession(string modelPath, bool useGpu)
    {
        Log("\n=== ONNX Runtime 진단 시작 ===");
        Log($"모델 경로: {modelPath}");
        Log($"GPU 요청: {useGpu}");
        Log($"ONNX Runtime 버전: {typeof(InferenceSession).Assembly.GetName().Version}");

        // 1. 파일 존재 확인
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"모델 파일을 찾을 수 없습니다: {modelPath}");
        }

        var fileInfo = new FileInfo(modelPath);
        Log($"✓ 모델 파일 존재 확인 (크기: {fileInfo.Length / 1024 / 1024} MB)");

        var so = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC,
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE
        };

        string deviceMode = "CPU";

        if (useGpu)
        {
            Log("\n--- GPU 모드 시도 ---");
            try
            {
                // DirectML 또는 CUDA 사용 가능 여부 확인
                var providers = OrtEnv.Instance().GetAvailableProviders();
                Log($"사용 가능한 Providers: {string.Join(", ", providers)}");

                // DirectML 우선 시도
                if (providers.Contains("DmlExecutionProvider"))
                {
                    Log("✓ DirectML Provider 사용 가능");
                    so.AppendExecutionProvider_DML(0);
                    Log("✓ DirectML Provider 추가됨");
                    deviceMode = "GPU (DirectML)";
                }
                // CUDA가 있으면 CUDA 사용
                else if (providers.Contains("CUDAExecutionProvider"))
                {
                    Log("✓ CUDA Provider 사용 가능");
                    so.AppendExecutionProvider_CUDA(0);
                    Log("✓ CUDA Provider 추가됨");
                    deviceMode = "GPU (CUDA)";
                }
                else
                {
                    Log("⚠️ GPU Provider를 사용할 수 없습니다!");
                    Log("\n--- CPU 모드로 Fallback ---");
                    deviceMode = "CPU (GPU Provider 없음)";
                }
            }
            catch (Exception ex)
            {
                Log($"\n❌ GPU 초기화 실패!");
                Log($"에러 타입: {ex.GetType().Name}");
                Log($"에러 메시지: {ex.Message}");

                if (ex.InnerException != null)
                {
                    Log($"내부 에러: {ex.InnerException.Message}");
                }

                Log($"\n전체 스택 트레이스:");
                Log(ex.ToString());
                Log("\n--- CPU 모드로 Fallback ---");
                deviceMode = "CPU (GPU 실패)";
            }
        }
        else
        {
            Log("\n--- CPU 모드 ---");
        }

        InferenceSession session;
        try
        {
            Log($"\n모델 로딩 시도 중 ({deviceMode})...");
            session = new InferenceSession(modelPath, so);
            Log($"✅ 모델 로드 성공!");
        }
        catch (Exception ex)
        {
            Log($"\n❌ 모델 로드 실패!");
            Log($"에러: {ex.Message}");

            if (ex.InnerException != null)
            {
                Log($"내부 에러: {ex.InnerException.Message}");
            }

            Log($"\n가능한 원인:");
            Log("  1. 모델 파일이 손상됨");
            Log("  2. ONNX Runtime 버전과 모델 버전 불일치");
            Log("  3. 메모리 부족");

            Log($"\n전체 에러:");
            Log(ex.ToString());

            throw;
        }

        Log("\n=== 진단 완료 ===");
        Log($"최종 모드: {deviceMode}");
        Log($"입력: {string.Join(", ", session.InputMetadata.Keys)}");
        Log($"출력: {string.Join(", ", session.OutputMetadata.Keys)}");
        Log("");

        return (session, deviceMode);
    }

    /// <summary>
    /// [Debug 모드 전용] x64 Debug 빌드 경로 기준으로 고정된 상대 경로를 사용하여 모델을 찾습니다.
    /// (bin/x64/Debug/net8.0-windows -> ../../../../../models)
    /// </summary>
    public static string? FindModelInDebug(string filename)
    {
#if DEBUG
        string baseDir = AppDomain.CurrentDomain.BaseDirectory;

        // 현재 위치: .../WpfAiRunner/bin/x64/Debug/net8.0-windows/
        // 목표 위치: .../models/
        string fixedPath = Path.Combine(baseDir, "..", "..", "..", "..", "..", "models", filename);

        string fullPath = Path.GetFullPath(fixedPath); // 경로 정규화
        if (File.Exists(fullPath))
        {
            Log($"[DEBUG] 모델 자동 검색 성공: {fullPath}");
            return fullPath;
        }
        else
        {
            Log($"[DEBUG] 모델을 찾을 수 없음: {fullPath}");
        }
#endif
        return null;
    }
}
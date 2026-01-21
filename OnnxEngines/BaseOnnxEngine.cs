using Microsoft.ML.OnnxRuntime;

namespace OnnxEngines
{
    public abstract class BaseOnnxEngine : IDisposable
    {
        // 자식들이 사용할 세션 (protected)
        protected InferenceSession? _session;
        public string DeviceMode { get; protected set; } = "CPU";

        public BaseOnnxEngine() { }

        public BaseOnnxEngine(string modelPath, bool useGpu)
        {
            LoadModel(modelPath, useGpu);
        }

        public virtual void LoadModel(string modelPath, bool useGpu)
        {
            // 기존 세션 정리
            _session?.Dispose();

            var options = new SessionOptions();
            DeviceMode = "CPU";

            if (useGpu)
            {
                try
                {
                    options.AppendExecutionProvider_DML(0);
                    DeviceMode = "GPU (DirectML)";
                }
                catch
                {
                    DeviceMode = "CPU (Fallback)";
                }
            }

            try
            {
                _session = new InferenceSession(modelPath, options);

                // [핵심] GPU 모드일 때만 웜업 호출 (템플릿 메서드 패턴)
                if (DeviceMode.Contains("GPU"))
                {
                    OnWarmup();
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to load model: {ex.Message}", ex);
            }
        }

        // 자식이 구현할 웜업 로직 (기본은 빈 껍데기)
        protected virtual void OnWarmup() { }

        // Dispose 패턴 구현
        public virtual void Dispose()
        {
            _session?.Dispose();
            _session = null;
            GC.SuppressFinalize(this);
        }
    }
}
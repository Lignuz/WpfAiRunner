using HPPH.SkiaSharp;
using StableDiffusion.NET;

namespace SdGgufEngine
{
    public class ModelParameter
    {
        public string Prompt { get; set; } = "";
        public string NegativePrompt { get; set; } = "";
        public int Width { get; set; } = 512;
        public int Height { get; set; } = 512;
        public int Steps { get; set; } = 20;
        public float Cfg { get; set; } = 7.5f;
        public float Guidance { get; set; } = 1.0f;
        public long Seed { get; set; } = -1; // -1일 경우 랜덤 시드
    }

    public class SdEngine : IDisposable
    {
        private DiffusionModel? _model;
        public Action<string>? LogAction { get; set; }
        // 수정: 3번째 인자를 이미지(byte[]) 대신 진행률(float)로 변경
        public Action<int, int, float>? ProgressAction { get; set; }

        public void Initialize()
        {
            StableDiffusionCpp.InitializeEvents();
            StableDiffusionCpp.Log += (s, e) => LogAction?.Invoke(e.Text);

            // 에러 발생 지점 수정: e.Image 제거 및 e.Progress(float) 전달
            StableDiffusionCpp.Progress += (s, e) =>
            {
                ProgressAction?.Invoke(e.Step, e.Steps, (float)e.Progress);
            };
        }

        public bool LoadModel(string modelPath, bool useGpu)
        {
            try
            {
                _model?.Dispose();
                var parameter = DiffusionModelParameter.Create().WithModelPath(modelPath);

                if (!useGpu) parameter.WithMultithreading();

                _model = new DiffusionModel(parameter);
                return true;
            }
            catch (Exception ex)
            {
                LogAction?.Invoke($"Error: {ex.Message}");
                return false;
            }
        }

        public byte[]? Generate(ModelParameter param)
        {
            if (_model == null) return null;

            var genParam = ImageGenerationParameter.TextToImage(param.Prompt)
                .WithNegativePrompt(param.NegativePrompt)
                .WithSize(param.Width, param.Height)
                .WithSteps(param.Steps)
                .WithCfg(param.Cfg)
                .WithGuidance(param.Guidance)
                .WithSeed(param.Seed)
                .WithSampler(Sampler.Euler);

            var image = _model.GenerateImage(genParam);
            return image?.ToPng();
        }

        public void Dispose() => _model?.Dispose();
    }
}
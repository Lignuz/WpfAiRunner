using HPPH;
using HPPH.SkiaSharp;
using SkiaSharp;
using StableDiffusion.NET;

namespace SdGgufEngine
{
    public enum GenerationMode
    {
        TextToImage,
        ImageToImage,
        Inpainting
    }

    public class ModelParameter
    {
        public GenerationMode Mode { get; set; } = GenerationMode.TextToImage;
        public string Prompt { get; set; } = "";
        public string NegativePrompt { get; set; } = "";
        public int Width { get; set; } = 512;
        public int Height { get; set; } = 512;
        public int Steps { get; set; } = 20;
        public float Cfg { get; set; } = 7.5f;
        public long Seed { get; set; } = -1;
        public byte[]? InputImage { get; set; }
        public byte[]? MaskImage { get; set; }
        public float Strength { get; set; } = 0.75f;
    }

    public class SdEngine : IDisposable
    {
        private DiffusionModel? _model;
        private bool _isInitialized = false;

        public Action<string>? LogAction { get; set; }
        public Action<int, int, float>? ProgressAction { get; set; }

        public void Initialize()
        {
            if (_isInitialized) return;

            StableDiffusionCpp.InitializeEvents();
            StableDiffusionCpp.Log += (s, e) => LogAction?.Invoke(e.Text);
            StableDiffusionCpp.Progress += (s, e) => ProgressAction?.Invoke(e.Step, e.Steps, (float)e.Progress);
            _isInitialized = true;
        }

        public bool LoadModel(string modelPath, bool useGpu)
        {
            try
            {
                _model?.Dispose();
                var parameter = DiffusionModelParameter.Create()
                    .WithModelPath(modelPath)
                    .WithFlashAttention();

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

            IImage? inputImg = null;
            IImage? maskImg = null;
            IImage? resultImg = null;

            try
            {
                // 1. 해상도 정보 보관 및 이미지 로드
                int originalW = 512, originalH = 512;
                if (param.InputImage != null)
                {
                    using var ms = new MemoryStream(param.InputImage);
                    using var nativeBmp = SKBitmap.Decode(ms);
                    originalW = nativeBmp.Width;
                    originalH = nativeBmp.Height;
                }

                inputImg = param.InputImage != null ? LoadImageFromBytes(param.InputImage, param.Width, param.Height) : null;
                maskImg = param.MaskImage != null ? LoadImageFromBytes(param.MaskImage, param.Width, param.Height) : null;

                // 2. 파라미터 구성
                var genParam = ImageGenerationParameter.TextToImage(param.Prompt)
                    .WithNegativePrompt(param.NegativePrompt)
                    .WithSize(param.Width, param.Height)
                    .WithSteps(param.Steps)
                    .WithCfg(param.Cfg)
                    .WithSeed(param.Seed)
                    .WithSampler(Sampler.Euler);

                if (param.Mode == GenerationMode.Inpainting && inputImg != null && maskImg != null)
                {
                    genParam.WithInitImage(inputImg)
                            .WithMaskImage(maskImg)
                            .WithStrength(Math.Max(param.Strength, 0.75f)); // 최소 0.75 이상 강제
                }
                else if (param.Mode == GenerationMode.ImageToImage && inputImg != null)
                {
                    genParam.WithInitImage(inputImg).WithStrength(param.Strength);
                }

                resultImg = _model.GenerateImage(genParam);
                if (resultImg == null) return null;

                // 3. 결과 복원 (Upscale)
                byte[] png512 = resultImg.ToPng();
                using var resStream = new MemoryStream(png512);
                using var resBmp = SKBitmap.Decode(resStream);
                using var finalBmp = resBmp.Resize(new SKImageInfo(originalW, originalH), SKFilterQuality.High);
                using var finalImg = SKImage.FromBitmap(finalBmp);
                using var data = finalImg.Encode(SKEncodedImageFormat.Png, 100);
                return data.ToArray();
            }
            finally
            {
                (inputImg as IDisposable)?.Dispose();
                (maskImg as IDisposable)?.Dispose();
                (resultImg as IDisposable)?.Dispose();
            }
        }

        private IImage LoadImageFromBytes(byte[] imageBytes, int targetW, int targetH)
        {
            using var stream = new MemoryStream(imageBytes);
            using var skBitmap = SKBitmap.Decode(stream);
            using var resized = skBitmap.Resize(new SKImageInfo(targetW, targetH), SKFilterQuality.High);
            using var skImage = SKImage.FromBitmap(resized);
            return skImage.ToImage();
        }

        public void Dispose() => _model?.Dispose();
    }
}
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using OnnxEngines.Utils;

namespace OnnxEngines.Rmbg;

public class RmbgEngine : BaseOnnxEngine
{
    private const int ModelSize = 1024;

    protected override void OnWarmup()
    {
        if (_session == null) return; // 안전장치

        try
        {
            var dummyTensor = new DenseTensor<float>(new[] { 1, 3, ModelSize, ModelSize });
            string inputName = _session.InputMetadata.Keys.First();
            using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, dummyTensor) });
        }
        catch { }
    }

    public byte[] RemoveBackground(byte[] imageBytes, float threshold = 0.0f, SKColor? bgColor = null)
    {
        if (_session == null) throw new InvalidOperationException("Model not loaded.");

        using var srcImage = SKBitmap.Decode(imageBytes).Copy(SKColorType.Rgba8888);
        int originalW = srcImage.Width;
        int originalH = srcImage.Height;

        var inputTensor = srcImage.ToTensor(ModelSize, ModelSize);

        var inputName = _session.InputMetadata.Keys.First();
        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) });
        var outputTensor = results.First().AsTensor<float>();

        // 마스크 생성
        using var maskImage = new SKBitmap(ModelSize, ModelSize, SKColorType.Gray8, SKAlphaType.Opaque);
        Span<byte> maskPixels = maskImage.GetPixelSpan();

        // 병렬 처리로 속도 최적화 가능하지만 안전하게 루프 사용
        for (int i = 0; i < maskPixels.Length; i++)
        {
            // Tensor는 NCHW [1, 1, 1024, 1024] -> Flat하게 접근 가능
            float val = outputTensor.GetValue(i);
            if (val < threshold) val = 0;
            maskPixels[i] = (byte)Math.Clamp(val * 255, 0, 255);
        }

        // 마스크 리사이즈 (High Quality)
        using var resizedMask = maskImage.Resize(
            new SKImageInfo(originalW, originalH, SKColorType.Gray8, SKAlphaType.Opaque),
            new SKSamplingOptions(SKCubicResampler.Mitchell));

        // [중요 수정] 결과 이미지를 Unpremul(Straight Alpha)로 생성
        // 이렇게 해야 R,G,B 값을 그대로 유지한 채 Alpha만 변경해도 Skia가 색상을 왜곡하지 않습니다.
        using var resultImage = new SKBitmap(new SKImageInfo(originalW, originalH, SKColorType.Rgba8888, SKAlphaType.Unpremul));

        ReadOnlySpan<byte> srcSpan = srcImage.GetPixelSpan();
        ReadOnlySpan<byte> maskSpan = resizedMask.GetPixelSpan();
        Span<byte> resSpan = resultImage.GetPixelSpan();

        int bpp = 4;

        for (int y = 0; y < originalH; y++)
        {
            int rowOff = y * originalW; // Gray8은 rowBytes == width (보통)
            int srcRowOff = y * srcImage.RowBytes;
            int resRowOff = y * resultImage.RowBytes;

            for (int x = 0; x < originalW; x++)
            {
                int srcIdx = srcRowOff + x * bpp;
                int resIdx = resRowOff + x * bpp;

                // 마스크 값
                byte maskVal = maskSpan[rowOff + x];

                byte r = srcSpan[srcIdx];
                byte g = srcSpan[srcIdx + 1];
                byte b = srcSpan[srcIdx + 2];

                if (bgColor.HasValue)
                {
                    // 배경색 합성은 직접 계산하므로 결과는 완전 불투명(255)이 됨
                    float alpha = maskVal / 255.0f;
                    var bg = bgColor.Value;

                    resSpan[resIdx] = (byte)(r * alpha + bg.Red * (1 - alpha));
                    resSpan[resIdx + 1] = (byte)(g * alpha + bg.Green * (1 - alpha));
                    resSpan[resIdx + 2] = (byte)(b * alpha + bg.Blue * (1 - alpha));
                    resSpan[resIdx + 3] = 255;
                }
                else
                {
                    // 투명 배경: Straight Alpha 방식 (RGB 유지, A만 변경)
                    // SKAlphaType.Unpremul 덕분에 이 방식이 올바르게 작동함
                    resSpan[resIdx] = r;
                    resSpan[resIdx + 1] = g;
                    resSpan[resIdx + 2] = b;
                    resSpan[resIdx + 3] = maskVal;
                }
            }
        }

        using var ms = new MemoryStream();
        using var data = resultImage.Encode(SKEncodedImageFormat.Png, 100);
        data.SaveTo(ms);
        return ms.ToArray();
    }
}
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using OnnxEngines.Utils;

namespace OnnxEngines.Upscaling;

public class RealEsrganEngine : BaseOnnxEngine
{
    private const int ModelInputSize = 128;
    private const int Overlap = 14;
    private const int StepSize = ModelInputSize - (Overlap * 2);

    protected override void OnWarmup()
    {
        if (_session == null) return; // 안전장치

        try
        {
            var dummyTensor = new DenseTensor<float>(new[] { 1, 3, ModelInputSize, ModelInputSize });
            string inputName = _session.InputMetadata.Keys.First();
            using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, dummyTensor) });
        }
        catch { }
    }

    public byte[] Upscale(byte[] imageBytes, IProgress<double>? progress = null)
    {
        if (_session == null) throw new InvalidOperationException("Model not loaded.");

        using var srcImage = SKBitmap.Decode(imageBytes).Copy(SKColorType.Rgba8888);
        int w = srcImage.Width;
        int h = srcImage.Height;

        int padW = w + (Overlap * 2);
        int padH = h + (Overlap * 2);

        // 패딩 이미지 생성 (Draw Image with Offset)
        using var paddedSrc = new SKBitmap(padW, padH, SKColorType.Rgba8888, SKAlphaType.Premul);
        using (var canvas = new SKCanvas(paddedSrc))
        {
            // 기본은 Transparent (0,0,0,0)이나 Black일 수 있음. 필요시 Clear 호출
            canvas.Clear(SKColors.Black);
            canvas.DrawBitmap(srcImage, Overlap, Overlap);
        }

        int outW = w * 4;
        int outH = h * 4;
        using var resultImage = new SKBitmap(outW, outH, SKColorType.Rgba8888, SKAlphaType.Premul);
        using var resultCanvas = new SKCanvas(resultImage);

        int countX = (int)Math.Ceiling((double)w / StepSize);
        int countY = (int)Math.Ceiling((double)h / StepSize);
        int totalTiles = countX * countY;
        int processedCount = 0;

        for (int y = 0; y < countY; y++)
        {
            for (int x = 0; x < countX; x++)
            {
                int srcX = x * StepSize;
                int srcY = y * StepSize;

                if (srcX + ModelInputSize > padW) srcX = padW - ModelInputSize;
                if (srcY + ModelInputSize > padH) srcY = padH - ModelInputSize;

                // 타일 잘라내기
                using var tile = new SKBitmap(ModelInputSize, ModelInputSize);
                paddedSrc.ExtractSubset(tile, SKRectI.Create(srcX, srcY, ModelInputSize, ModelInputSize));

                // 추론
                using var upscaledTile = ProcessTile(tile);

                int destX = x * StepSize * 4;
                int destY = y * StepSize * 4;
                int cropX = Overlap * 4;
                int cropY = Overlap * 4;
                int cropW = StepSize * 4;
                int cropH = StepSize * 4;

                if (destX + cropW > outW) destX = outW - cropW;
                if (destY + cropH > outH) destY = outH - cropH;

                // 유효 영역 잘라내기
                var validRect = SKRectI.Create(cropX, cropY, cropW, cropH);
                using var validPart = new SKBitmap(cropW, cropH);
                upscaledTile.ExtractSubset(validPart, validRect);

                // 결과 캔버스에 그리기
                resultCanvas.DrawBitmap(validPart, destX, destY);

                processedCount++;
                progress?.Report((double)processedCount / totalTiles);
            }
        }

        using var ms = new MemoryStream();
        using var data = resultImage.Encode(SKEncodedImageFormat.Png, 100);
        data.SaveTo(ms);
        return ms.ToArray();
    }

    private SKBitmap ProcessTile(SKBitmap tileImage)
    {
        // TensorHelper 사용
        var inputTensor = tileImage.ToTensor(ModelInputSize, ModelInputSize);

        var inputName = _session!.InputMetadata.Keys.First();
        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) });
        var outputTensor = results.First().AsTensor<float>();

        int outSize = ModelInputSize * 4;
        var outputImage = new SKBitmap(outSize, outSize, SKColorType.Rgba8888, SKAlphaType.Premul);
        Span<byte> pixels = outputImage.GetPixelSpan();
        int bpp = outputImage.BytesPerPixel;

        for (int y = 0; y < outSize; y++)
        {
            int rowOff = y * outputImage.RowBytes;
            for (int x = 0; x < outSize; x++)
            {
                // 모델 출력은 보통 0~1 사이 값
                float r = Math.Clamp(outputTensor[0, 0, y, x], 0, 1) * 255;
                float g = Math.Clamp(outputTensor[0, 1, y, x], 0, 1) * 255;
                float b = Math.Clamp(outputTensor[0, 2, y, x], 0, 1) * 255;

                int i = rowOff + (x * bpp);
                pixels[i] = (byte)r;
                pixels[i + 1] = (byte)g;
                pixels[i + 2] = (byte)b;
                pixels[i + 3] = 255;
            }
        }

        return outputImage;
    }
}
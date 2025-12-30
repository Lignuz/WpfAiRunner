using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using OnnxEngines.Utils;

namespace OnnxEngines.Upscaling;

public class RealEsrganEngine : IDisposable
{
    private InferenceSession? _session;
    public string DeviceMode { get; private set; } = "CPU";

    private const int ModelInputSize = 128;
    private const int Overlap = 14;
    private const int StepSize = ModelInputSize - (Overlap * 2);

    public void LoadModel(string modelPath, bool useGpu)
    {
        _session?.Dispose();
        // OnnxHelper 사용
        (_session, DeviceMode) = OnnxHelper.LoadSession(modelPath, useGpu);

        if (DeviceMode == "GPU")
        {
            try
            {
                // 간단한 웜업
                var dummyTensor = new DenseTensor<float>(new[] { 1, 3, ModelInputSize, ModelInputSize });
                string inputName = _session.InputMetadata.Keys.First();
                using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, dummyTensor) });
            }
            catch { }
        }
    }

    public byte[] Upscale(byte[] imageBytes, IProgress<double>? progress = null)
    {
        if (_session == null) throw new InvalidOperationException("Model not loaded.");

        using var srcImage = Image.Load<Rgba32>(imageBytes);
        int w = srcImage.Width;
        int h = srcImage.Height;

        int padW = w + (Overlap * 2);
        int padH = h + (Overlap * 2);

        using var paddedSrc = new Image<Rgba32>(padW, padH);
        paddedSrc.Mutate(x => x.DrawImage(srcImage, new Point(Overlap, Overlap), 1f));

        int outW = w * 4;
        int outH = h * 4;
        using var resultImage = new Image<Rgba32>(outW, outH);

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

                using var tile = paddedSrc.Clone(ctx => ctx.Crop(new Rectangle(srcX, srcY, ModelInputSize, ModelInputSize)));
                using var upscaledTile = ProcessTile(tile);

                int destX = x * StepSize * 4;
                int destY = y * StepSize * 4;
                int cropX = Overlap * 4;
                int cropY = Overlap * 4;
                int cropW = StepSize * 4;
                int cropH = StepSize * 4;

                if (destX + cropW > outW) destX = outW - cropW;
                if (destY + cropH > outH) destY = outH - cropH;

                var validRect = new Rectangle(cropX, cropY, cropW, cropH);
                using var validPart = upscaledTile.Clone(ctx => ctx.Crop(validRect));
                resultImage.Mutate(ctx => ctx.DrawImage(validPart, new Point(destX, destY), 1f));

                processedCount++;
                progress?.Report((double)processedCount / totalTiles);
            }
        }

        using var ms = new MemoryStream();
        resultImage.SaveAsPng(ms);
        return ms.ToArray();
    }

    private Image<Rgba32> ProcessTile(Image<Rgba32> tileImage)
    {
        // TensorHelper 사용 (0~1 정규화)
        var inputTensor = tileImage.ToTensor(ModelInputSize, ModelInputSize);

        var inputName = _session!.InputMetadata.Keys.First();
        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) });
        var outputTensor = results.First().AsTensor<float>();

        int outSize = ModelInputSize * 4;
        var outputImage = new Image<Rgba32>(outSize, outSize);
        outputImage.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < outSize; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < outSize; x++)
                {
                    float r = Math.Clamp(outputTensor[0, 0, y, x], 0, 1) * 255;
                    float g = Math.Clamp(outputTensor[0, 1, y, x], 0, 1) * 255;
                    float b = Math.Clamp(outputTensor[0, 2, y, x], 0, 1) * 255;
                    row[x] = new Rgba32((byte)r, (byte)g, (byte)b);
                }
            }
        });

        return outputImage;
    }

    public void Dispose() => _session?.Dispose();
}
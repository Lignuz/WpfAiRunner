using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using OnnxEngines.Utils;
using SamEngine;

namespace OnnxEngines.Sam;

public class SamSegmenter : ISamSegmenter
{
    private InferenceSession? _encoderSession;
    private InferenceSession? _decoderSession;

    // SAM 모델의 표준 입력 해상도 (1024x1024)
    private const int TargetSize = 1024;
    private float[]? _imageEmbeddings;
    private int _orgW, _orgH;
    private int _resizedW, _resizedH;
    private Tensor<float>? _lastMaskTensor;
    public string DeviceMode { get; private set; } = "CPU";

    public void LoadModels(string encoderPath, string decoderPath, bool useGpu)
    {
        (_encoderSession, DeviceMode) = OnnxHelper.LoadSession(encoderPath, useGpu);
        (_decoderSession, _) = OnnxHelper.LoadSession(decoderPath, useGpu);
    }

    public void EncodeImage(byte[] imageBytes)
    {
        if (_encoderSession == null) throw new InvalidOperationException("Encoder not loaded.");
        using var image = SKBitmap.Decode(imageBytes).Copy(SKColorType.Rgba8888);
        _orgW = image.Width; _orgH = image.Height;

        float scale = (float)TargetSize / Math.Max(_orgW, _orgH);
        _resizedW = (int)(_orgW * scale); _resizedH = (int)(_orgH * scale);

        using var resized = image.Resize(new SKImageInfo(_resizedW, _resizedH), new SKSamplingOptions(SKCubicResampler.Mitchell));
        using var paddedImage = new SKBitmap(TargetSize, TargetSize, SKColorType.Rgba8888, SKAlphaType.Premul);
        using (var canvas = new SKCanvas(paddedImage))
        {
            canvas.Clear(SKColors.Black);
            canvas.DrawBitmap(resized, 0, 0);
        }

        string inputName = _encoderSession.InputMetadata.Keys.First();
        var inputTensor = CreateEncoderInputTensor(paddedImage);
        using var results = _encoderSession.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) });
        var outputTensor = results.First().AsTensor<float>();
        _imageEmbeddings = ConvertEmbeddingsToCHW(outputTensor);
    }

    public (List<float> Scores, byte[] BestMaskBytes, int BestIndex) Predict(float x, float y)
    {
        if (_decoderSession == null || _imageEmbeddings == null) return (new List<float>(), Array.Empty<byte>(), -1);

        var embedTensor = new DenseTensor<float>(_imageEmbeddings, new[] { 1, 256, 64, 64 });
        var pointCoords = new DenseTensor<float>(new[] { 1, 2, 2 });
        float scale = (float)TargetSize / Math.Max(_orgW, _orgH);
        pointCoords[0, 0, 0] = x * scale; pointCoords[0, 0, 1] = y * scale;
        pointCoords[0, 1, 0] = 0f; pointCoords[0, 1, 1] = 0f;

        var pointLabels = new DenseTensor<float>(new[] { 1, 2 });
        pointLabels[0, 0] = 1.0f; pointLabels[0, 1] = -1.0f;

        var maskInput = new DenseTensor<float>(new[] { 1, 1, 256, 256 });
        var hasMaskInput = new DenseTensor<float>(new[] { 0.0f }, new[] { 1 });
        var origImSize = new DenseTensor<float>(new[] { (float)TargetSize, (float)TargetSize }, new[] { 2 });

        var inputs = new List<NamedOnnxValue> {
            NamedOnnxValue.CreateFromTensor("image_embeddings", embedTensor),
            NamedOnnxValue.CreateFromTensor("point_coords", pointCoords),
            NamedOnnxValue.CreateFromTensor("point_labels", pointLabels),
            NamedOnnxValue.CreateFromTensor("mask_input", maskInput),
            NamedOnnxValue.CreateFromTensor("has_mask_input", hasMaskInput),
            NamedOnnxValue.CreateFromTensor("orig_im_size", origImSize),
        };

        using var decResults = _decoderSession.Run(inputs);
        var maskResult = decResults.FirstOrDefault(r => r.Name == "masks") ?? decResults.First();
        var iouResult = decResults.FirstOrDefault(r => r.Name == "iou_predictions");
        _lastMaskTensor = maskResult.AsTensor<float>().ToDenseTensor();
        var iouTensor = iouResult?.AsTensor<float>();

        var scores = new List<float>();
        int candidateCount = _lastMaskTensor.Dimensions[1];
        int bestIndex = 0; float maxScore = -1f;
        for (int i = 0; i < candidateCount; i++)
        {
            float rawScore = iouTensor != null ? iouTensor[0, i] : 0.0f;
            if (rawScore > 1.0f) rawScore = 1.0f; if (rawScore < 0.0f) rawScore = 0.0f;
            scores.Add(rawScore);
            if (rawScore > maxScore) { maxScore = rawScore; bestIndex = i; }
        }
        byte[] bestMaskBytes = MaskTensorToPng(_lastMaskTensor, bestIndex);
        return (scores, bestMaskBytes, bestIndex);
    }

    public byte[] GetMaskImage(int index)
    {
        if (_lastMaskTensor == null) return Array.Empty<byte>();
        return MaskTensorToPng(_lastMaskTensor, index);
    }

    private DenseTensor<float> CreateEncoderInputTensor(SKBitmap img1024)
    {
        const int H = 1024; const int W = 1024; const int C = 3;
        var tensor = new DenseTensor<float>(new[] { H, W, C });

        ReadOnlySpan<byte> pixels = img1024.GetPixelSpan();
        int bpp = img1024.BytesPerPixel; // 4

        for (int y = 0; y < H; y++)
        {
            int rowOff = y * img1024.RowBytes;
            for (int x = 0; x < W; x++)
            {
                int i = rowOff + (x * bpp);
                tensor[y, x, 0] = (pixels[i] - 123.675f) / 58.395f;     // R
                tensor[y, x, 1] = (pixels[i + 1] - 116.28f) / 57.12f; // G
                tensor[y, x, 2] = (pixels[i + 2] - 103.53f) / 57.375f; // B
            }
        }
        return tensor;
    }

    private float[] ConvertEmbeddingsToCHW(Tensor<float> outputTensor)
    {
        var dims = outputTensor.Dimensions.ToArray();
        var data = outputTensor.ToArray();
        var chw = new float[256 * 64 * 64];
        if (dims.Length == 4)
        {
            if (dims[1] == 256) { Buffer.BlockCopy(data, 0, chw, 0, sizeof(float) * chw.Length); return chw; }
            if (dims[3] == 256)
            {
                int h = 64, w = 64, c = 256;
                Parallel.For(0, h * w, i =>
                {
                    int y = i / w; int x = i % w;
                    int baseSrc = (y * w + x) * c;
                    int baseDst = y * w + x;
                    for (int ch = 0; ch < c; ch++) chw[(ch * h * w) + baseDst] = data[baseSrc + ch];
                });
                return chw;
            }
        }
        if (chw.Length == data.Length) Buffer.BlockCopy(data, 0, chw, 0, sizeof(float) * chw.Length);
        return chw;
    }

    private byte[] MaskTensorToPng(Tensor<float> maskTensor, int maskIndex)
    {
        int rank = maskTensor.Dimensions.Length;
        int h = 256, w = 256;
        if (rank >= 2) { h = maskTensor.Dimensions[rank - 2]; w = maskTensor.Dimensions[rank - 1]; }

        using var rawMask = new SKBitmap(w, h, SKColorType.Gray8, SKAlphaType.Opaque);
        Span<byte> pixels = rawMask.GetPixelSpan();

        for (int y = 0; y < h; y++)
        {
            int rowOff = y * rawMask.RowBytes;
            for (int x = 0; x < w; x++)
            {
                float v = 0f;
                if (rank == 4) v = maskTensor[0, maskIndex, y, x];
                else if (rank == 3) v = maskTensor[maskIndex, y, x];
                else v = maskTensor[y, x];
                pixels[rowOff + x] = v > 0.0f ? (byte)255 : (byte)0;
            }
        }

        double ratioW = (double)_resizedW / TargetSize;
        double ratioH = (double)_resizedH / TargetSize;
        int validW = (int)(w * ratioW); int validH = (int)(h * ratioH);
        validW = Math.Clamp(validW, 1, w); validH = Math.Clamp(validH, 1, h);

        using var cropped = new SKBitmap(validW, validH);
        rawMask.ExtractSubset(cropped, SKRectI.Create(0, 0, validW, validH));

        using var final = cropped.Resize(new SKImageInfo(_orgW, _orgH), new SKSamplingOptions(SKCubicResampler.Mitchell));

        using var ms = new MemoryStream();
        using var data = final.Encode(SKEncodedImageFormat.Png, 100);
        data.SaveTo(ms);
        return ms.ToArray();
    }

    public void Dispose()
    {
        _encoderSession?.Dispose();
        _decoderSession?.Dispose();
    }
}
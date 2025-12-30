using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Collections.Concurrent;

namespace SamEngine;

// [수정] ISamSegmenter 인터페이스 구현
public class Sam2Segmenter : ISamSegmenter
{
    private InferenceSession? _encoderSession;
    private InferenceSession? _decoderSession;

    private const int TargetSize = 1024;

    private List<NamedOnnxValue>? _encoderResults;

    private int _orgW, _orgH;
    private int _resizedW, _resizedH;
    private Tensor<float>? _lastMaskTensor;

    public string DeviceMode { get; private set; } = "CPU";

    public void LoadModels(string encoderPath, string decoderPath, bool useGpu)
    {
        var so = new SessionOptions();
        if (useGpu)
        {
            try { so.AppendExecutionProvider_CUDA(0); DeviceMode = "GPU"; }
            catch { DeviceMode = "CPU"; }
        }

        _encoderSession = new InferenceSession(encoderPath, so);
        _decoderSession = new InferenceSession(decoderPath, so);
    }

    public void EncodeImage(byte[] imageBytes)
    {
        if (_encoderSession == null) throw new InvalidOperationException("Encoder not loaded.");

        using var image = Image.Load<Rgba32>(imageBytes);
        _orgW = image.Width;
        _orgH = image.Height;

        float scale = (float)TargetSize / Math.Max(_orgW, _orgH);
        _resizedW = (int)(_orgW * scale);
        _resizedH = (int)(_orgH * scale);

        image.Mutate(x => x.Resize(_resizedW, _resizedH));
        using var paddedImage = new Image<Rgba32>(TargetSize, TargetSize);
        paddedImage.Mutate(x => x.BackgroundColor(Color.Black));
        paddedImage.Mutate(x => x.DrawImage(image, new Point(0, 0), 1f));

        var inputTensor = CreateEncoderInputTensor(paddedImage);
        string inputName = _encoderSession.InputMetadata.Keys.First();

        var results = _encoderSession.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) });

        _encoderResults = new List<NamedOnnxValue>();
        foreach (var r in results)
        {
            var tensor = r.AsTensor<float>().ToDenseTensor();
            _encoderResults.Add(NamedOnnxValue.CreateFromTensor(r.Name, tensor));
        }
    }

    public (List<float> Scores, byte[] BestMaskBytes, int BestIndex) Predict(float x, float y)
    {
        if (_decoderSession == null || _encoderResults == null)
            return (new List<float>(), Array.Empty<byte>(), -1);

        var inputs = new List<NamedOnnxValue>();
        inputs.AddRange(_encoderResults);

        var pointCoords = new DenseTensor<float>(new[] { 1, 2, 2 });
        float scale = (float)TargetSize / Math.Max(_orgW, _orgH);
        pointCoords[0, 0, 0] = x * scale;
        pointCoords[0, 0, 1] = y * scale;
        pointCoords[0, 1, 0] = 0f;
        pointCoords[0, 1, 1] = 0f;

        var pointLabels = new DenseTensor<float>(new[] { 1, 2 });
        pointLabels[0, 0] = 1.0f;
        pointLabels[0, 1] = -1.0f;

        var maskInput = new DenseTensor<float>(new[] { 1, 1, 256, 256 });
        var hasMaskInput = new DenseTensor<float>(new[] { 0.0f }, new[] { 1 });

        inputs.Add(NamedOnnxValue.CreateFromTensor("point_coords", pointCoords));
        inputs.Add(NamedOnnxValue.CreateFromTensor("point_labels", pointLabels));
        inputs.Add(NamedOnnxValue.CreateFromTensor("mask_input", maskInput));
        inputs.Add(NamedOnnxValue.CreateFromTensor("has_mask_input", hasMaskInput));

        using var decResults = _decoderSession.Run(inputs);

        var maskResult = decResults.FirstOrDefault(r => r.Name == "masks") ?? decResults.First();
        var iouResult = decResults.FirstOrDefault(r => r.Name == "iou_predictions");

        _lastMaskTensor = maskResult.AsTensor<float>().ToDenseTensor();

        var iouTensor = iouResult?.AsTensor<float>();
        var scores = new List<float>();
        int bestIndex = 0;
        float maxScore = -1f;
        int candidateCount = _lastMaskTensor.Dimensions[1];

        for (int i = 0; i < candidateCount; i++)
        {
            float rawScore = iouTensor != null ? iouTensor[0, i] : 0.0f;
            if (rawScore > 1.0f) rawScore = 1.0f;
            if (rawScore < 0.0f) rawScore = 0.0f;
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

    private DenseTensor<float> CreateEncoderInputTensor(Image<Rgba32> img1024)
    {
        const int H = 1024;
        const int W = 1024;
        var tensor = new DenseTensor<float>(new[] { 1, 3, H, W });

        img1024.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < H; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < W; x++)
                {
                    var p = row[x];
                    tensor[0, 0, y, x] = (p.R - 123.675f) / 58.395f;
                    tensor[0, 1, y, x] = (p.G - 116.28f) / 57.12f;
                    tensor[0, 2, y, x] = (p.B - 103.53f) / 57.375f;
                }
            }
        });
        return tensor;
    }

    private byte[] MaskTensorToPng(Tensor<float> maskTensor, int maskIndex)
    {
        int h = 256, w = 256;
        using var rawMask = new Image<L8>(w, h);

        rawMask.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    float v = maskTensor[0, maskIndex, y, x];
                    float probability = 1.0f / (1.0f + MathF.Exp(-v));
                    row[x] = new L8((byte)(probability * 255));
                }
            }
        });

        double ratioW = (double)_resizedW / TargetSize;
        double ratioH = (double)_resizedH / TargetSize;
        int validW = (int)(w * ratioW);
        int validH = (int)(h * ratioH);
        validW = Math.Clamp(validW, 1, w);
        validH = Math.Clamp(validH, 1, h);

        rawMask.Mutate(x => x.Crop(new Rectangle(0, 0, validW, validH)));
        rawMask.Mutate(x => x.Resize(new ResizeOptions
        {
            Size = new Size(_orgW, _orgH),
            Mode = ResizeMode.Stretch,
            Sampler = KnownResamplers.Bicubic
        }));

        using var ms = new MemoryStream();
        rawMask.SaveAsPng(ms);
        return ms.ToArray();
    }

    public void Dispose()
    {
        _encoderSession?.Dispose();
        _decoderSession?.Dispose();
    }
}
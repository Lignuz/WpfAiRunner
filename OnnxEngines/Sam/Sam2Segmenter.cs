using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using OnnxEngines.Utils;
using SamEngine;

namespace OnnxEngines.Sam;

public class Sam2Segmenter : BaseOnnxEngine, ISamSegmenter
{
    private InferenceSession? _encoderSession;
    private InferenceSession? _decoderSession;
    private const int TargetSize = 1024;
    private List<NamedOnnxValue>? _encoderResults;
    private int _orgW, _orgH;
    private int _resizedW, _resizedH;
    private Tensor<float>? _lastMaskTensor;

    // SAM2용 정규화 값
    private readonly float[] _mean = new[] { 123.675f, 116.28f, 103.53f };
    private readonly float[] _std = new[] { 58.395f, 57.12f, 57.375f };

    public void LoadModels(string encoderPath, string decoderPath, bool useGpu)
    {
        (_encoderSession, DeviceMode) = OnnxHelper.LoadSession(encoderPath, useGpu);
        (_decoderSession, _) = OnnxHelper.LoadSession(decoderPath, useGpu);
    }

    public void EncodeImage(byte[] imageBytes)
    {
        if (_encoderSession == null) throw new InvalidOperationException("Encoder not loaded.");

        using var image = SKBitmap.Decode(imageBytes).Copy(SKColorType.Rgba8888);
        _orgW = image.Width;
        _orgH = image.Height;

        float scale = (float)TargetSize / Math.Max(_orgW, _orgH);
        _resizedW = (int)(_orgW * scale);
        _resizedH = (int)(_orgH * scale);

        // 1. 리사이즈
        using var resizedImage = image.Resize(new SKImageInfo(_resizedW, _resizedH), new SKSamplingOptions(SKCubicResampler.Mitchell));

        // 2. 패딩 (Black Background)
        using var paddedImage = new SKBitmap(TargetSize, TargetSize, SKColorType.Rgba8888, SKAlphaType.Premul);
        using (var canvas = new SKCanvas(paddedImage))
        {
            canvas.Clear(SKColors.Black);
            canvas.DrawBitmap(resizedImage, 0, 0);
        }

        // TensorHelper 사용 (Mean/Std 포함)
        var inputTensor = paddedImage.ToTensor(TargetSize, TargetSize, _mean, _std);

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
        int bestIndex = 0; float maxScore = -1f;
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

    private byte[] MaskTensorToPng(Tensor<float> maskTensor, int maskIndex)
    {
        int h = 256, w = 256;
        // Gray8 포맷 사용 (1채널)
        using var rawMask = new SKBitmap(w, h, SKColorType.Gray8, SKAlphaType.Opaque);
        Span<byte> pixels = rawMask.GetPixelSpan();

        for (int y = 0; y < h; y++)
        {
            int rowOff = y * rawMask.RowBytes;
            for (int x = 0; x < w; x++)
            {
                float v = maskTensor[0, maskIndex, y, x];
                float probability = 1.0f / (1.0f + MathF.Exp(-v));
                pixels[rowOff + x] = (byte)(probability * 255);
            }
        }

        // 유효 영역 계산 및 크롭
        double ratioW = (double)_resizedW / TargetSize;
        double ratioH = (double)_resizedH / TargetSize;
        int validW = (int)(w * ratioW);
        int validH = (int)(h * ratioH);
        validW = Math.Clamp(validW, 1, w);
        validH = Math.Clamp(validH, 1, h);

        using var croppedMask = new SKBitmap(validW, validH);
        // ExtractSubset은 픽셀 메모리를 공유하거나 복사본을 생성할 수 있음. 
        // Resize 전에 새로운 비트맵 객체가 필요하므로 ExtractSubset 사용.
        rawMask.ExtractSubset(croppedMask, SKRectI.Create(0, 0, validW, validH));

        // 최종 원본 크기로 리사이즈
        using var finalMask = croppedMask.Resize(new SKImageInfo(_orgW, _orgH), new SKSamplingOptions(SKCubicResampler.Mitchell));

        using var ms = new MemoryStream();
        using var data = finalMask.Encode(SKEncodedImageFormat.Png, 100);
        data.SaveTo(ms);
        return ms.ToArray();
    }

    public override void Dispose()
    {
        _encoderSession?.Dispose();
        _decoderSession?.Dispose();
    }
}
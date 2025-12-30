using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using OnnxEngines.Utils;
using SamEngine;

namespace OnnxEngines.Sam;

// ISamSegmenter 인터페이스 구현
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
        using var image = Image.Load<Rgba32>(imageBytes);
        _orgW = image.Width; _orgH = image.Height;
        float scale = (float)TargetSize / Math.Max(_orgW, _orgH);
        _resizedW = (int)(_orgW * scale); _resizedH = (int)(_orgH * scale);
        image.Mutate(x => x.Resize(_resizedW, _resizedH));
        using var paddedImage = new Image<Rgba32>(TargetSize, TargetSize);
        paddedImage.Mutate(x => x.BackgroundColor(Color.Black));
        paddedImage.Mutate(x => x.DrawImage(image, new Point(0, 0), 1f));
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

    // 이 모델은 HWC 텐서 생성 로직을 유지합니다.
    private DenseTensor<float> CreateEncoderInputTensor(Image<Rgba32> img1024)
    {
        const int H = 1024; const int W = 1024; const int C = 3;
        var tensor = new DenseTensor<float>(new[] { H, W, C });
        img1024.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < H; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < W; x++)
                {
                    var p = row[x];
                    tensor[y, x, 0] = (p.R - 123.675f) / 58.395f;
                    tensor[y, x, 1] = (p.G - 116.28f) / 57.12f;
                    tensor[y, x, 2] = (p.B - 103.53f) / 57.375f;
                }
            }
        });
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
        using var rawMask = new Image<L8>(w, h);
        rawMask.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    float v = 0f;
                    if (rank == 4) v = maskTensor[0, maskIndex, y, x];
                    else if (rank == 3) v = maskTensor[maskIndex, y, x];
                    else v = maskTensor[y, x];
                    row[x] = new L8(v > 0.0f ? (byte)255 : (byte)0);
                }
            }
        });
        double ratioW = (double)_resizedW / TargetSize;
        double ratioH = (double)_resizedH / TargetSize;
        int validW = (int)(w * ratioW); int validH = (int)(h * ratioH);
        validW = Math.Clamp(validW, 1, w); validH = Math.Clamp(validH, 1, h);
        rawMask.Mutate(x => x.Crop(new Rectangle(0, 0, validW, validH)));
        rawMask.Mutate(x => x.Resize(new ResizeOptions { Size = new Size(_orgW, _orgH), Mode = ResizeMode.Stretch }));
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
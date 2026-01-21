using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using OnnxEngines.Utils;

namespace OnnxEngines.Depth;

public class DepthEstimator : BaseOnnxEngine
{
    private const int ModelSize = 518;

    private Tensor<float>? _lastOutputTensor;
    private int _lastOrigW, _lastOrigH;

    public DepthEstimator(string modelPath, bool useGpu = false) : base(modelPath, useGpu) { }

    protected override void OnWarmup()
    {
        if (_session == null) return; // 안전장치

        try
        {
            var dummyTensor = new DenseTensor<float>(new[] { 1, 3, ModelSize, ModelSize });
            string inputName = _session.InputMetadata.Keys.First();
            using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, dummyTensor) });
        }
        catch {}
    }

    // 1단계: 추론만 수행
    public void RunInference(byte[] imageBytes)
    {
        if (_session == null) throw new System.InvalidOperationException("Model not loaded.");

        using var src = SKBitmap.Decode(imageBytes).Copy(SKColorType.Rgba8888);
        _lastOrigW = src.Width;
        _lastOrigH = src.Height;

        // TensorHelper 사용
        var inputTensor = src.ToTensor(ModelSize, ModelSize);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image", inputTensor)
        };

        if (_session.InputMetadata.Count > 0)
        {
            string inputName = _session.InputMetadata.Keys.First();
            inputs[0] = NamedOnnxValue.CreateFromTensor(inputName, inputTensor);
        }

        using var results = _session.Run(inputs);
        var outputRaw = results.First().AsTensor<float>();

        _lastOutputTensor = outputRaw.ToDenseTensor();
    }

    // 2단계: 저장된 결과로 스타일만 적용
    public byte[] GetDepthMap(ColormapStyle style)
    {
        if (_lastOutputTensor == null)
            throw new InvalidOperationException("Inference has not been run yet.");

        // 캐시된 텐서를 사용하여 이미지 생성
        using var outputImg = TensorToColorMap(_lastOutputTensor, ModelSize, ModelSize, style);

        // 원본 크기 복원
        // Skia의 Resize는 새 객체를 반환하므로 outputImg를 리사이즈한 결과를 저장
        using var resizedImg = outputImg.Resize(new SKImageInfo(_lastOrigW, _lastOrigH), new SKSamplingOptions(SKCubicResampler.Mitchell));
        using var data = resizedImg.Encode(SKEncodedImageFormat.Png, 100);
        return data.ToArray();
    }

    private SKBitmap TensorToColorMap(Tensor<float> tensor, int w, int h, ColormapStyle style)
    {
        float min = float.MaxValue;
        float max = float.MinValue;

        foreach (var val in tensor)
        {
            if (val < min) min = val;
            if (val > max) max = val;
        }

        float range = max - min;
        if (range < 0.00001f) range = 1f;

        var img = new SKBitmap(w, h, SKColorType.Rgba8888, SKAlphaType.Premul);
        Span<byte> pixels = img.GetPixelSpan();
        int bytesPerPixel = img.BytesPerPixel;

        for (int y = 0; y < h; y++)
        {
            int rowOffset = y * img.RowBytes;
            for (int x = 0; x < w; x++)
            {
                float val = tensor[0, y, x];
                float norm = (val - min) / range;
                SKColor color = ColorMapper.GetColor(norm, style);

                int offset = rowOffset + (x * bytesPerPixel);
                pixels[offset] = color.Red;
                pixels[offset + 1] = color.Green;
                pixels[offset + 2] = color.Blue;
                pixels[offset + 3] = 255;
            }
        }

        return img;
    }
}
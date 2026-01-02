using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using OnnxEngines.Utils;

namespace OnnxEngines.Depth;

public class DepthEstimator : IDisposable
{
    private readonly InferenceSession _session;
    private const int ModelSize = 518;

    // 마지막 추론 결과를 저장할 변수 (캐시)
    private Tensor<float>? _lastOutputTensor;
    private int _lastOrigW, _lastOrigH;

    public string DeviceMode { get; private set; } = "CPU";

    public DepthEstimator(string modelPath, bool useGpu = false)
    {
        // OnnxHelper 사용
        (_session, DeviceMode) = OnnxHelper.LoadSession(modelPath, useGpu);

        if (DeviceMode.Contains("GPU"))
        {
            try
            {
                // Warm-up
                var dummyTensor = new DenseTensor<float>(new[] { 1, 3, ModelSize, ModelSize });
                string inputName = _session.InputMetadata.Keys.First();
                using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, dummyTensor) });
            }
            catch { }
        }
    }

    // 1단계: 추론만 수행 (Heavy)
    public void RunInference(byte[] imageBytes)
    {
        using var src = Image.Load<Rgba32>(imageBytes);
        _lastOrigW = src.Width;
        _lastOrigH = src.Height;

        // TensorHelper 사용 (리사이즈 + 정규화 통합 처리)
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

        // 결과를 캐시 변수에 깊은 복사(ToDenseTensor)로 저장
        _lastOutputTensor = outputRaw.ToDenseTensor();
    }

    // 2단계: 저장된 결과로 스타일만 적용 (Light)
    public byte[] GetDepthMap(ColormapStyle style)
    {
        if (_lastOutputTensor == null)
            throw new InvalidOperationException("Inference has not been run yet.");

        // 캐시된 텐서를 사용하여 이미지 생성
        using var outputImg = TensorToColorMap(_lastOutputTensor, ModelSize, ModelSize, style);

        // 원본 크기 복원
        outputImg.Mutate(ctx => ctx.Resize(_lastOrigW, _lastOrigH));

        using var ms = new MemoryStream();
        outputImg.SaveAsPng(ms);
        return ms.ToArray();
    }

    private Image<Rgba32> TensorToColorMap(Tensor<float> tensor, int w, int h, ColormapStyle style)
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

        var img = new Image<Rgba32>(w, h);
        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    float val = tensor[0, y, x];
                    float norm = (val - min) / range;
                    row[x] = ColorMapper.GetColor(norm, style);
                }
            }
        });

        return img;
    }

    public void Dispose() => _session?.Dispose();
}
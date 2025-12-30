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

    public byte[] ProcessImage(byte[] imageBytes)
    {
        using var src = Image.Load<Rgba32>(imageBytes);
        int origW = src.Width;
        int origH = src.Height;

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
        var outputTensor = results.First().AsTensor<float>();

        using var outputImg = TensorToGrayscale(outputTensor, ModelSize, ModelSize);
        outputImg.Mutate(ctx => ctx.Resize(origW, origH));

        using var ms = new MemoryStream();
        outputImg.SaveAsPng(ms);
        return ms.ToArray();
    }

    private Image<L8> TensorToGrayscale(Tensor<float> tensor, int w, int h)
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

        var img = new Image<L8>(w, h);
        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    float val = tensor[0, y, x];
                    float norm = (val - min) / range;
                    row[x] = new L8((byte)(norm * 255f));
                }
            }
        });

        return img;
    }

    public void Dispose() => _session?.Dispose();
}
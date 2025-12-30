using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace DepthEngine;

public class DepthEstimator : IDisposable
{
    private readonly InferenceSession _session;
    private const int ModelSize = 518; // Depth Anything V2
    public string DeviceMode { get; private set; } = "CPU";

    public DepthEstimator(string modelPath, bool useGpu = false)
    {
        using var so = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC,
        };

        if (useGpu)
        {
            try
            {
                so.AppendExecutionProvider_CUDA(0);
                DeviceMode = "GPU (CUDA)";
            }
            catch
            {
                DeviceMode = "CPU (Fallback)";
            }
        }

        _session = new InferenceSession(modelPath, so);

        // ▼▼▼ [추가] GPU 모드일 경우 미리 한 번 돌려서 예열(Warm-up) ▼▼▼
        if (DeviceMode.Contains("GPU"))
        {
            RunWarmup();
        }
    }

    // ▼▼▼ [추가] 웜업 메서드 ▼▼▼
    private void RunWarmup()
    {
        try
        {
            // 가짜 데이터 생성 (1, 3, 518, 518)
            var dummyTensor = new DenseTensor<float>(new[] { 1, 3, ModelSize, ModelSize });

            // 입력 이름 찾기
            string inputName = "image";
            if (_session.InputMetadata.Count > 0)
                inputName = _session.InputMetadata.Keys.First();

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, dummyTensor)
            };

            // 추론 실행 (결과는 버림) - 이때 GPU 초기화 완료됨
            using var results = _session.Run(inputs);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Warmup failed: {ex.Message}");
        }
    }

    public byte[] ProcessImage(byte[] imageBytes)
    {
        // (기존 코드와 동일)
        using var src = Image.Load<Rgba32>(imageBytes);
        int origW = src.Width;
        int origH = src.Height;

        using var inputImg = src.Clone(ctx => ctx.Resize(ModelSize, ModelSize));

        var inputTensor = new DenseTensor<float>(new[] { 1, 3, ModelSize, ModelSize });
        inputImg.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < ModelSize; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < ModelSize; x++)
                {
                    inputTensor[0, 0, y, x] = row[x].R / 255f;
                    inputTensor[0, 1, y, x] = row[x].G / 255f;
                    inputTensor[0, 2, y, x] = row[x].B / 255f;
                }
            }
        });

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
        // (기존 코드와 동일)
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
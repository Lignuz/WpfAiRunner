using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using OnnxEngines.Utils;
using System.Runtime.InteropServices;

namespace OnnxEngines.Style;

public class AnimeGanEngine : IDisposable
{
    private InferenceSession? _session;
    public string DeviceMode { get; private set; } = "None";

    public void LoadModel(string modelPath, bool useGpu)
    {
        _session?.Dispose();

        // GPU 드라이버 충돌(887A0006) 방지를 위한 옵션 설정
        var options = new SessionOptions();

        if (useGpu)
        {
            try
            {
                // DmlFusedNode 오류 방지를 위해 최적화 끄기
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
                options.AppendExecutionProvider_DML(0);
                DeviceMode = "GPU (DML - NoOpt)";
            }
            catch
            {
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                DeviceMode = "CPU (Fallback)";
            }
        }
        else
        {
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            DeviceMode = "CPU";
        }

        _session = new InferenceSession(modelPath, options);
    }

    public byte[] Process(byte[] imageBytes)
    {
        if (_session == null) throw new InvalidOperationException("Model not loaded.");

        using var srcBitmap = SKBitmap.Decode(imageBytes);
        using var image = srcBitmap.Copy(SKColorType.Rgba8888);

        // 1. 크기 조정 (32의 배수)
        int w = image.Width - (image.Width % 32);
        int h = image.Height - (image.Height % 32);

        if (w <= 0 || h <= 0) w = h = 32;

        using var resizedImage = (w != image.Width || h != image.Height)
            ? image.Resize(new SKImageInfo(w, h, SKColorType.Rgba8888), new SKSamplingOptions(SKCubicResampler.Mitchell))
            : image.Copy(SKColorType.Rgba8888);

        // 2. 텐서 변환
        // DenseTensor로 선언해야 Buffer 속성에 접근 가능
        var inputTensor = new DenseTensor<float>(new[] { 1, h, w, 3 });
        var tensorSpan = inputTensor.Buffer.Span;

        ReadOnlySpan<byte> pixels = resizedImage.GetPixelSpan();
        int bpp = resizedImage.BytesPerPixel;

        for (int y = 0; y < h; y++)
        {
            int rowOff = y * resizedImage.RowBytes;
            int tensorRowOff = y * w * 3;

            for (int x = 0; x < w; x++)
            {
                int pIdx = rowOff + (x * bpp);
                int tIdx = tensorRowOff + (x * 3);

                // NHWC 포맷
                tensorSpan[tIdx] = (pixels[pIdx] / 127.5f) - 1.0f;     // R
                tensorSpan[tIdx + 1] = (pixels[pIdx + 1] / 127.5f) - 1.0f; // G
                tensorSpan[tIdx + 2] = (pixels[pIdx + 2] / 127.5f) - 1.0f; // B
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(), inputTensor)
        };

        // 3. 추론
        using var results = _session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // [수정] Tensor<float>는 Buffer가 없으므로 DenseTensor<float>로 캐스팅
        if (outputTensor is not DenseTensor<float> denseOutput)
        {
            // 만약 DenseTensor가 아니라면 복사해서 변환
            denseOutput = outputTensor.ToDenseTensor();
        }

        // 4. 후처리
        using var outputImage = new SKBitmap(new SKImageInfo(w, h, SKColorType.Rgba8888, SKAlphaType.Opaque));
        Span<byte> outPixels = outputImage.GetPixelSpan();

        // 캐스팅된 denseOutput 사용
        var outSpan = denseOutput.Buffer.Span;

        for (int y = 0; y < h; y++)
        {
            int rowOff = y * outputImage.RowBytes;
            int tensorRowOff = y * w * 3;

            for (int x = 0; x < w; x++)
            {
                int tIdx = tensorRowOff + (x * 3);

                float r = (outSpan[tIdx] + 1.0f) * 127.5f;
                float g = (outSpan[tIdx + 1] + 1.0f) * 127.5f;
                float b = (outSpan[tIdx + 2] + 1.0f) * 127.5f;

                int pIdx = rowOff + (x * bpp);
                outPixels[pIdx] = (byte)Math.Clamp(r, 0, 255);
                outPixels[pIdx + 1] = (byte)Math.Clamp(g, 0, 255);
                outPixels[pIdx + 2] = (byte)Math.Clamp(b, 0, 255);
                outPixels[pIdx + 3] = 255;
            }
        }

        using var ms = new MemoryStream();
        using var data = outputImage.Encode(SKEncodedImageFormat.Png, 100);
        data.SaveTo(ms);
        return ms.ToArray();
    }

    public void Dispose() => _session?.Dispose();
}
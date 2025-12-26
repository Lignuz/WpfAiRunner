using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace LamaEngine;

public class LamaInpainter : IDisposable
{
    private readonly InferenceSession _session;
    private const int ModelSize = 512;
    public string DeviceMode { get; private set; } = "CPU";

    public LamaInpainter(string modelPath, bool useGpu = false)
    {
        var so = new SessionOptions
        {
            // 1. 기본 설정 (CPU 기준)
            // NOTE: ORT_ENABLE_BASIC is conservative and tends to be stable across many models.
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC,
            IntraOpNumThreads = Math.Max(1, Environment.ProcessorCount - 1),
            InterOpNumThreads = 1
        };

        if (useGpu)
        {
            // 2. GPU 가속 시도 
            try
            {
                // CUDA(0번 GPU) 사용을 시도합니다.
                // 주의: GPU 패키지가 설치되어 있고, 드라이버가 깔려 있어야 성공합니다.
                so.AppendExecutionProvider_CUDA(0);
                DeviceMode = "GPU (CUDA)";
            }
            catch (Exception)
            {
                // 3. 실패 시 조용히 넘어감 (Fallback)
                // 여기에 도달했다는 것은:
                // - NVIDIA 그래픽카드가 없거나
                // - CUDA 드라이버가 설치되지 않았거나
                // - 호환되지 않는 GPU임
                // => 이 경우 SessionOptions는 기본값(CPU) 상태를 유지합니다.
                System.Diagnostics.Debug.WriteLine("GPU Acceleration failed. Switching to CPU.");
                DeviceMode = "CPU (Fallback)";
            }
        }
        else
        {
            DeviceMode = "CPU";
        }

        // 4. 세션 생성 (GPU가 성공했으면 GPU로, 실패했으면 CPU로 로드됨)
        _session = new InferenceSession(modelPath, so);

        if (DeviceMode.Contains("GPU"))
        {
            RunWarmup();
        }
    }

    private void RunWarmup()
    {
        try
        {
            // 1. 가짜 데이터 생성 (512x512, 0으로 채움)
            // ImageSharp를 쓰지 않고 텐서를 직접 만들어서 빠르게 처리
            var dummyImage = new DenseTensor<float>(new[] { 1, 3, ModelSize, ModelSize });
            var dummyMask = new DenseTensor<float>(new[] { 1, 1, ModelSize, ModelSize });

            var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image", dummyImage),
            NamedOnnxValue.CreateFromTensor("mask", dummyMask)
        };

            // 2. 추론 실행 (결과는 버림)
            // 이 과정에서 GPU 초기화, 메모리 할당, 커널 컴파일이 모두 완료됨
            using var results = _session.Run(inputs);
        }
        catch (Exception ex)
        {
            // 웜업 실패는 무시해도 됨 (실제 실행 때 다시 시도할 테니)
            System.Diagnostics.Debug.WriteLine($"Warmup failed: {ex.Message}");
        }
    }

    public byte[] ProcessImage(byte[] imageBytes, byte[] maskBytes)
    {
        using var src = Image.Load<Rgba32>(imageBytes);
        using var mask = Image.Load<L8>(maskBytes);

        // If mask is empty -> no-op (return original image).
        if (!HasAnyMask(mask))
        {
            using var msNoop = new MemoryStream();
            src.SaveAsPng(msNoop);
            return msNoop.ToArray();
        }

        var roi = GetMaskBoundingBox(mask);
        roi = AdjustRoiToSquare(roi, src.Width, src.Height);

        using var srcCrop = src.Clone(ctx => ctx.Crop(roi));
        using var maskCrop = mask.Clone(ctx => ctx.Crop(roi));

        using var src512 = srcCrop.Clone(ctx => ctx.Resize(ModelSize, ModelSize, KnownResamplers.Bicubic));
        using var mask512 = maskCrop.Clone(ctx => ctx.Resize(ModelSize, ModelSize, KnownResamplers.NearestNeighbor));

        using var out512 = RunInference(src512, mask512);

        out512.Mutate(ctx => ctx.Resize(roi.Width, roi.Height, KnownResamplers.Bicubic));

        src.Mutate(ctx => ctx.DrawImage(out512, new Point(roi.X, roi.Y), 1f));

        using var ms = new MemoryStream();
        src.SaveAsPng(ms);
        return ms.ToArray();
    }

    private Image<Rgba32> RunInference(Image<Rgba32> img, Image<L8> mask)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image", MakeImageTensor(img)),
            NamedOnnxValue.CreateFromTensor("mask", MakeMaskTensor(mask))
        };

        using var results = _session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        return TensorToImageAuto(outputTensor);
    }

    
    private bool HasAnyMask(Image<L8> mask)
    {
        bool any = false;
        mask.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < mask.Height && !any; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < mask.Width; x++)
                {
                    if (row[x].PackedValue > 127)
                    {
                        any = true;
                        break;
                    }
                }
            }
        });
        return any;
    }

    private Rectangle GetMaskBoundingBox(Image<L8> mask)
    {
        int minX = mask.Width, minY = mask.Height, maxX = 0, maxY = 0;
        bool found = false;

        mask.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < mask.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < mask.Width; x++)
                {
                    if (row[x].PackedValue > 127)
                    {
                        if (x < minX) minX = x;
                        if (x > maxX) maxX = x;
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                        found = true;
                    }
                }
            }
        });

        if (!found) return new Rectangle(0, 0, mask.Width, mask.Height);
        return new Rectangle(minX, minY, maxX - minX + 1, maxY - minY + 1);
    }

    private Rectangle AdjustRoiToSquare(Rectangle roi, int imgW, int imgH)
    {
        int cx = roi.X + roi.Width / 2;
        int cy = roi.Y + roi.Height / 2;

        int size = Math.Max(roi.Width, roi.Height);
        size = (int)(size * 2.0); // Context 확보
        size = Math.Max(size, 512);

        int half = size / 2;
        int x = cx - half;
        int y = cy - half;

        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x + size > imgW) x = imgW - size;
        if (y + size > imgH) y = imgH - size;

        if (x < 0 || y < 0) return new Rectangle(0, 0, imgW, imgH);
        return new Rectangle(x, y, size, size);
    }

    private DenseTensor<float> MakeImageTensor(Image<Rgba32> img)
    {
        var t = new DenseTensor<float>(new[] { 1, 3, ModelSize, ModelSize });
        img.ProcessPixelRows(a =>
        {
            for (int y = 0; y < ModelSize; y++)
            {
                var row = a.GetRowSpan(y);
                for (int x = 0; x < ModelSize; x++)
                {
                    t[0, 0, y, x] = row[x].R / 255f;
                    t[0, 1, y, x] = row[x].G / 255f;
                    t[0, 2, y, x] = row[x].B / 255f;
                }
            }
        });
        return t;
    }

    private DenseTensor<float> MakeMaskTensor(Image<L8> mask)
    {
        var t = new DenseTensor<float>(new[] { 1, 1, ModelSize, ModelSize });
        mask.ProcessPixelRows(a =>
        {
            for (int y = 0; y < ModelSize; y++)
            {
                var row = a.GetRowSpan(y);
                for (int x = 0; x < ModelSize; x++)
                {
                    t[0, 0, y, x] = row[x].PackedValue > 127 ? 1f : 0f;
                }
            }
        });
        return t;
    }

    // --- Auto Scale Logic (흰색 화면 방지) ---
    private Image<Rgba32> TensorToImageAuto(Tensor<float> outTensor)
    {
        var (minV, maxV) = SampleMinMax(outTensor);
        var scale = DecideOutputScale(minV, maxV);

        float to01_mul = scale switch
        {
            OutputScale.Minus1To1 => 0.5f,
            OutputScale.ZeroTo255 => 1f / 255f,
            _ => 1f
        };

        var img = new Image<Rgba32>(ModelSize, ModelSize);
        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < ModelSize; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < ModelSize; x++)
                {
                    // NCHW
                    float r = outTensor[0, 0, y, x];
                    float g = outTensor[0, 1, y, x];
                    float b = outTensor[0, 2, y, x];
                    
                    row[x] = new Rgba32(
                        ToByte(r, scale, to01_mul),
                        ToByte(g, scale, to01_mul),
                        ToByte(b, scale, to01_mul),
                        255
                    );
                }
            }
        });
        return img;
    }

    private (float minV, float maxV) SampleMinMax(Tensor<float> t)
    {
        if (t is DenseTensor<float> dt)
        {
            var span = dt.Buffer.Span;
            int step = Math.Max(1, span.Length / 1000);
            float minV = float.PositiveInfinity;
            float maxV = float.NegativeInfinity;
            for (int i = 0; i < span.Length; i += step)
            {
                float v = span[i];
                if (v < minV) minV = v;
                if (v > maxV) maxV = v;
            }
            if (float.IsInfinity(minV)) return (0, 1);
            return (minV, maxV);
        }
        return (0, 1);
    }

    private enum OutputScale { ZeroTo1, ZeroTo255, Minus1To1 }

    private OutputScale DecideOutputScale(float minV, float maxV)
    {
        if (minV < -0.1f && maxV <= 1.5f) return OutputScale.Minus1To1;
        if (maxV > 2.0f) return OutputScale.ZeroTo255;
        return OutputScale.ZeroTo1;
    }

    private byte ToByte(float v, OutputScale scale, float to01_mul)
    {
        if (float.IsNaN(v)) v = 0;
        if (scale == OutputScale.Minus1To1) v = (v + 1f) * to01_mul;
        else if (scale == OutputScale.ZeroTo255) v = v * to01_mul;
        v = Math.Clamp(v, 0f, 1f);
        return (byte)(v * 255f);
    }

    public void Dispose() => _session?.Dispose();
}
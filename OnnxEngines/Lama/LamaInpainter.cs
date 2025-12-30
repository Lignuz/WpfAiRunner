using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using OnnxEngines.Utils;

namespace OnnxEngines.Lama;

public class LamaInpainter : IDisposable
{
    private readonly InferenceSession _session;
    private const int ModelSize = 512;
    public string DeviceMode { get; private set; } = "CPU";

    public LamaInpainter(string modelPath, bool useGpu = false)
    {
        // OnnxHelper를 사용하여 세션 로드 및 디바이스 모드 설정
        (_session, DeviceMode) = OnnxHelper.LoadSession(modelPath, useGpu);

        // GPU 모드일 경우 웜업 실행
        if (DeviceMode == "GPU")
        {
            try
            {
                var dummyImage = new DenseTensor<float>(new[] { 1, 3, ModelSize, ModelSize });
                var dummyMask = new DenseTensor<float>(new[] { 1, 1, ModelSize, ModelSize });
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("image", dummyImage),
                    NamedOnnxValue.CreateFromTensor("mask", dummyMask)
                };
                using var results = _session.Run(inputs);
            }
            catch { }
        }
    }

    public byte[] ProcessImage(byte[] imageBytes, byte[] maskBytes)
    {
        using var src = Image.Load<Rgba32>(imageBytes);
        using var mask = Image.Load<L8>(maskBytes);

        // 마스크가 없으면 원본 그대로 반환
        if (!HasAnyMask(mask))
        {
            using var msNoop = new MemoryStream();
            src.SaveAsPng(msNoop);
            return msNoop.ToArray();
        }

        // 1. 마스크 영역(ROI) 계산 및 정사각형 보정
        var roi = GetMaskBoundingBox(mask);
        roi = AdjustRoiToSquare(roi, src.Width, src.Height);

        // 2. ROI 크롭
        using var srcCrop = src.Clone(ctx => ctx.Crop(roi));
        using var maskCrop = mask.Clone(ctx => ctx.Crop(roi));

        // 3. 추론 (512x512 리사이즈 및 복원은 내부에서 처리)
        using var out512 = RunInference(srcCrop, maskCrop);

        // 4. 결과물을 원래 ROI 크기로 복원
        out512.Mutate(ctx => ctx.Resize(roi.Width, roi.Height, KnownResamplers.Bicubic));

        // 5. 원본 이미지의 해당 위치에 덮어쓰기 (Inpainting 적용)
        src.Mutate(ctx => ctx.DrawImage(out512, new Point(roi.X, roi.Y), 1f));

        using var ms = new MemoryStream();
        src.SaveAsPng(ms);
        return ms.ToArray();
    }

    private Image<Rgba32> RunInference(Image<Rgba32> img, Image<L8> mask)
    {
        // TensorHelper를 사용하여 텐서 변환
        var imgTensor = img.ToTensor(ModelSize, ModelSize); // 0~1 정규화 포함
        var maskTensor = mask.ToMaskTensor(ModelSize, ModelSize); // 마스크 변환 포함

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image", imgTensor),
            NamedOnnxValue.CreateFromTensor("mask", maskTensor)
        };

        using var results = _session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // 후처리 (텐서 -> 이미지 변환, Auto Scale 적용)
        return TensorToImageAuto(outputTensor);
    }

    // --- Helper Methods ---

    private bool HasAnyMask(Image<L8> mask)
    {
        bool any = false;
        mask.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height && !any; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
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
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
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
        size = (int)(size * 2.0); // Context 확보를 위해 2배 확장
        size = Math.Max(size, 512);

        int half = size / 2;
        int x = cx - half;
        int y = cy - half;

        // 이미지 경계 벗어나지 않도록 보정
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x + size > imgW) x = imgW - size;
        if (y + size > imgH) y = imgH - size;

        if (x < 0 || y < 0) return new Rectangle(0, 0, imgW, imgH);
        return new Rectangle(x, y, size, size);
    }

    // --- Post-Processing Logic (Lama 모델 특성상 출력 스케일 자동 보정 필요) ---

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
            int step = Math.Max(1, span.Length / 1000); // 1000개 샘플링
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
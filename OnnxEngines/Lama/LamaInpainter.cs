using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using OnnxEngines.Utils;

namespace OnnxEngines.Lama;

public class LamaInpainter : BaseOnnxEngine
{
    private const int ModelSize = 512;

    public LamaInpainter(string modelPath, bool useGpu) : base(modelPath, useGpu) { }

    protected override void OnWarmup()
    {
        if (_session == null) return; // 안전장치

        try
        {
            // 웜업용 더미 데이터
            var dummy = new DenseTensor<float>(new[] { 1, 3, ModelSize, ModelSize });
            var dummyMask = new DenseTensor<float>(new[] { 1, 1, ModelSize, ModelSize });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image", dummy),
                NamedOnnxValue.CreateFromTensor("mask", dummyMask)
            };

            // [경고 해결] _session 뒤에 !를 붙여서 null이 아님을 명시 (LoadModel 호출 중이므로)
            using var res = _session!.Run(inputs);
        }
        catch
        {
            // 웜업 실패는 무시 (로그만 남기거나)
        }
    }

    public byte[] ProcessImage(byte[] imageBytes, byte[] maskBytes)
    {
        if (_session == null) throw new System.InvalidOperationException("Model not loaded.");

        // 마스크는 그레이스케일 등으로 읽기 위해 기본 디코드 사용
        using var src = SKBitmap.Decode(imageBytes).Copy(SKColorType.Rgba8888);
        using var mask = SKBitmap.Decode(maskBytes).Copy(SKColorType.Rgba8888); // 편의상 Rgba로 통일

        if (!HasAnyMask(mask))
        {
            // 변경 사항 없으면 그대로 저장
            using var dataNoop = src.Encode(SKEncodedImageFormat.Png, 100);
            return dataNoop.ToArray();
        }

        // 1. ROI 계산
        var roi = GetMaskBoundingBox(mask);
        roi = AdjustRoiToSquare(roi, src.Width, src.Height);

        // 2. ROI 크롭 (ExtractSubset은 픽셀 공유, 딥카피가 필요하면 Copy 사용)
        using var srcCrop = new SKBitmap(roi.Width, roi.Height);
        src.ExtractSubset(srcCrop, roi);

        using var maskCrop = new SKBitmap(roi.Width, roi.Height);
        mask.ExtractSubset(maskCrop, roi);

        // 3. 추론 (512x512 처리)
        using var out512 = RunInference(srcCrop, maskCrop);

        // 4. 결과물을 ROI 크기로 복원
        using var outResized = out512.Resize(new SKImageInfo(roi.Width, roi.Height), new SKSamplingOptions(SKCubicResampler.Mitchell));

        // 5. 원본 이미지에 덮어쓰기
        using var canvas = new SKCanvas(src);
        canvas.DrawBitmap(outResized, roi.Left, roi.Top);

        using var data = src.Encode(SKEncodedImageFormat.Png, 100);
        return data.ToArray();
    }

    private SKBitmap RunInference(SKBitmap img, SKBitmap mask)
    {
        if (_session == null) throw new System.InvalidOperationException("Model not loaded.");

        var imgTensor = img.ToTensor(ModelSize, ModelSize);
        var maskTensor = mask.ToMaskTensor(ModelSize, ModelSize);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image", imgTensor),
            NamedOnnxValue.CreateFromTensor("mask", maskTensor)
        };

        using var results = _session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        return TensorToImageAuto(outputTensor);
    }

    // --- Helper Methods ---

    private bool HasAnyMask(SKBitmap mask)
    {
        // 고속 픽셀 검사
        ReadOnlySpan<byte> pixels = mask.GetPixelSpan();
        int bpp = mask.BytesPerPixel; // 4 (Rgba8888)

        // Alpha 혹은 RGB 값 체크. Grayscale 변환된 경우 R=G=B
        for (int i = 0; i < pixels.Length; i += bpp)
        {
            // R채널(혹은 밝기)이 127보다 크면 마스크로 간주
            if (pixels[i] > 127) return true;
        }
        return false;
    }

    private SKRectI GetMaskBoundingBox(SKBitmap mask)
    {
        int minX = mask.Width, minY = mask.Height, maxX = 0, maxY = 0;
        bool found = false;

        ReadOnlySpan<byte> pixels = mask.GetPixelSpan();
        int width = mask.Width;
        int height = mask.Height;
        int rowBytes = mask.RowBytes;
        int bpp = mask.BytesPerPixel;

        for (int y = 0; y < height; y++)
        {
            int rowOff = y * rowBytes;
            for (int x = 0; x < width; x++)
            {
                if (pixels[rowOff + x * bpp] > 127)
                {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                    found = true;
                }
            }
        }

        if (!found) return new SKRectI(0, 0, width, height);
        // Width/Height 계산
        return SKRectI.Create(minX, minY, maxX - minX + 1, maxY - minY + 1);
    }

    private SKRectI AdjustRoiToSquare(SKRectI roi, int imgW, int imgH)
    {
        int cx = roi.MidX;
        int cy = roi.MidY;

        int size = Math.Max(roi.Width, roi.Height);
        size = (int)(size * 2.0);
        size = Math.Max(size, 512);

        int half = size / 2;
        int x = cx - half;
        int y = cy - half;

        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x + size > imgW) x = imgW - size;
        if (y + size > imgH) y = imgH - size;

        if (x < 0 || y < 0) return new SKRectI(0, 0, imgW, imgH);
        return SKRectI.Create(x, y, size, size);
    }

    // --- Post-Processing Logic ---

    private SKBitmap TensorToImageAuto(Tensor<float> outTensor)
    {
        var (minV, maxV) = SampleMinMax(outTensor);
        var scale = DecideOutputScale(minV, maxV);

        float to01_mul = scale switch
        {
            OutputScale.Minus1To1 => 0.5f,
            OutputScale.ZeroTo255 => 1f / 255f,
            _ => 1f
        };

        var img = new SKBitmap(ModelSize, ModelSize, SKColorType.Rgba8888, SKAlphaType.Premul);
        Span<byte> pixels = img.GetPixelSpan();
        int bpp = img.BytesPerPixel;

        for (int y = 0; y < ModelSize; y++)
        {
            int rowOff = y * img.RowBytes;
            for (int x = 0; x < ModelSize; x++)
            {
                float r = outTensor[0, 0, y, x];
                float g = outTensor[0, 1, y, x];
                float b = outTensor[0, 2, y, x];

                int offset = rowOff + (x * bpp);
                pixels[offset] = ToByte(r, scale, to01_mul);
                pixels[offset + 1] = ToByte(g, scale, to01_mul);
                pixels[offset + 2] = ToByte(b, scale, to01_mul);
                pixels[offset + 3] = 255;
            }
        }
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
}
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using OnnxEngines.Utils;
using System.Runtime.InteropServices;

namespace OnnxEngines.Colorization;

public class ColorizationEngine : IDisposable
{
    private InferenceSession? _session;
    public string DeviceMode { get; private set; } = "None";

    // DDColor 모델 입력 크기 (512x512 고정)
    private const int ModelInputSize = 512;

    // 1. LoadModel 메서드
    public void LoadModel(string modelPath, bool useGpu)
    {
        _session?.Dispose();
        (_session, DeviceMode) = OnnxHelper.LoadSession(modelPath, useGpu);
    }

    // 2. Process 메서드
    public byte[] Process(byte[] imageBytes)
    {
        if (_session == null) throw new InvalidOperationException("Model not loaded.");

        // 원본 이미지 로드 (Rgba8888 포맷 강제)
        using var originalImage = SKBitmap.Decode(imageBytes).Copy(SKColorType.Rgba8888);
        int origW = originalImage.Width;
        int origH = originalImage.Height;

        // ---------------------------------------------------------
        // 단계 1: 모델 입력 준비 (512x512)
        // ---------------------------------------------------------
        using var inputImage = originalImage.Resize(new SKImageInfo(ModelInputSize, ModelInputSize), new SKSamplingOptions(SKCubicResampler.Mitchell));

        var inputTensor = new DenseTensor<float>(new[] { 1, 3, ModelInputSize, ModelInputSize });

        // 픽셀 데이터 접근
        ReadOnlySpan<byte> inputPixels = inputImage.GetPixelSpan();
        int inputBytesPerPixel = inputImage.BytesPerPixel;

        for (int y = 0; y < ModelInputSize; y++)
        {
            int rowOffset = y * inputImage.RowBytes;
            for (int x = 0; x < ModelInputSize; x++)
            {
                int pIdx = rowOffset + (x * inputBytesPerPixel);

                float r = inputPixels[pIdx] / 255.0f;
                float g = inputPixels[pIdx + 1] / 255.0f;
                float b = inputPixels[pIdx + 2] / 255.0f;

                // RGB -> Lab 변환 후 L 채널만 추출
                RgbToLab(r, g, b, out float L, out _, out _);

                // L 채널만 있는 회색조 RGB로 변환
                LabToRgb(L, 0f, 0f, out float grayR, out float grayG, out float grayB);

                inputTensor[0, 0, y, x] = grayR;
                inputTensor[0, 1, y, x] = grayG;
                inputTensor[0, 2, y, x] = grayB;
            }
        }

        // ---------------------------------------------------------
        // 단계 2: 추론 (Inference)
        // ---------------------------------------------------------
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(), inputTensor) };
        using var results = _session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        bool isNchw = outputTensor.Dimensions[1] == 2;

        // ---------------------------------------------------------
        // 단계 3: 결과 합성 (블렌딩 & 원본 해상도 복원)
        // ---------------------------------------------------------
        using var outputImage = new SKBitmap(origW, origH, SKColorType.Rgba8888, SKAlphaType.Premul);

        ReadOnlySpan<byte> srcPixels = originalImage.GetPixelSpan();
        // 쓰기 가능한 Span 가져오기 (IntPtr을 통해 unsafe 접근 하거나 GetPixelSpan 사용)
        // SKBitmap.GetPixelSpan()은 8.0 이상에서 Span<byte>를 반환하여 쓰기가 가능합니다.
        Span<byte> dstPixels = outputImage.GetPixelSpan();

        int srcBytesPerPixel = originalImage.BytesPerPixel;
        int dstBytesPerPixel = outputImage.BytesPerPixel;

        for (int y = 0; y < origH; y++)
        {
            int srcRowOff = y * originalImage.RowBytes;
            int dstRowOff = y * outputImage.RowBytes;

            for (int x = 0; x < origW; x++)
            {
                // (A) 원본 픽셀에서 L(밝기) 추출
                int srcIdx = srcRowOff + (x * srcBytesPerPixel);
                float r = srcPixels[srcIdx] / 255.0f;
                float g = srcPixels[srcIdx + 1] / 255.0f;
                float b = srcPixels[srcIdx + 2] / 255.0f;

                RgbToLab(r, g, b, out float origL, out _, out _);

                // (B) 모델 출력에서 a, b 추출
                int modelX = (int)((float)x / origW * ModelInputSize);
                int modelY = (int)((float)y / origH * ModelInputSize);
                modelX = Math.Clamp(modelX, 0, ModelInputSize - 1);
                modelY = Math.Clamp(modelY, 0, ModelInputSize - 1);

                float predA, predB;
                if (isNchw)
                {
                    predA = outputTensor[0, 0, modelY, modelX];
                    predB = outputTensor[0, 1, modelY, modelX];
                }
                else
                {
                    predA = outputTensor[0, modelY, modelX, 0];
                    predB = outputTensor[0, modelY, modelX, 1];
                }

                // (C) Lab -> RGB 변환
                LabToRgb(origL, predA, predB, out float outR, out float outG, out float outB);

                int dstIdx = dstRowOff + (x * dstBytesPerPixel);
                dstPixels[dstIdx] = (byte)Math.Clamp(outR * 255f, 0, 255);     // R
                dstPixels[dstIdx + 1] = (byte)Math.Clamp(outG * 255f, 0, 255); // G
                dstPixels[dstIdx + 2] = (byte)Math.Clamp(outB * 255f, 0, 255); // B
                dstPixels[dstIdx + 3] = 255; // A
            }
        }

        using var data = outputImage.Encode(SKEncodedImageFormat.Png, 100);
        return data.ToArray();
    }

    public void Dispose() => _session?.Dispose();


    // ================================
    // OpenCV 스타일 색공간 변환 함수들
    // ================================

    private static void RgbToLab(float r, float g, float b, out float L, out float a, out float bb)
    {
        // sRGB -> Linear RGB
        r = SrgbToLinear(r);
        g = SrgbToLinear(g);
        b = SrgbToLinear(b);

        // Linear RGB -> XYZ (D65)
        float X = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
        float Y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
        float Z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

        // XYZ -> Lab
        const float Xn = 0.95047f;
        const float Yn = 1.00000f;
        const float Zn = 1.08883f;

        float fx = Fxyz(X / Xn);
        float fy = Fxyz(Y / Yn);
        float fz = Fxyz(Z / Zn);

        L = 116f * fy - 16f;
        a = 500f * (fx - fy);
        bb = 200f * (fy - fz);
    }

    private static void LabToRgb(float L, float a, float bb, out float r, out float g, out float b)
    {
        // Lab -> XYZ
        float fy = (L + 16f) / 116f;
        float fx = fy + (a / 500f);
        float fz = fy - (bb / 200f);

        const float Xn = 0.95047f;
        const float Yn = 1.00000f;
        const float Zn = 1.08883f;

        float X = Xn * Finv(fx);
        float Y = Yn * Finv(fy);
        float Z = Zn * Finv(fz);

        // XYZ -> Linear RGB
        float rl = X * 3.2404542f + Y * -1.5371385f + Z * -0.4985314f;
        float gl = X * -0.9692660f + Y * 1.8760108f + Z * 0.0415560f;
        float bl = X * 0.0556434f + Y * -0.2040259f + Z * 1.0572252f;

        // Linear RGB -> sRGB
        r = LinearToSrgb(rl);
        g = LinearToSrgb(gl);
        b = LinearToSrgb(bl);

        // Clamp
        r = Math.Clamp(r, 0f, 1f);
        g = Math.Clamp(g, 0f, 1f);
        b = Math.Clamp(b, 0f, 1f);
    }

    private static float SrgbToLinear(float c)
    {
        if (c <= 0.04045f) return c / 12.92f;
        return MathF.Pow((c + 0.055f) / 1.055f, 2.4f);
    }

    private static float LinearToSrgb(float c)
    {
        if (c <= 0.0031308f) return 12.92f * c;
        return 1.055f * MathF.Pow(c, 1f / 2.4f) - 0.055f;
    }

    private static float Fxyz(float t)
    {
        const float delta = 6f / 29f;
        const float delta3 = delta * delta * delta;
        if (t > delta3) return MathF.Pow(t, 1f / 3f);
        return (t / (3f * delta * delta)) + (4f / 29f);
    }

    private static float Finv(float ft)
    {
        const float delta = 6f / 29f;
        if (ft > delta) return ft * ft * ft;
        return 3f * delta * delta * (ft - 4f / 29f);
    }
}
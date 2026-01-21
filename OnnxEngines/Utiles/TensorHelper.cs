using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;

namespace OnnxEngines.Utils;

public static class TensorHelper
{
    /// <summary>
    /// SKBitmap 이미지를 NCHW [1, 3, H, W] 형태의 Tensor로 변환합니다.
    /// </summary>
    /// <param name="mean">Mean 정규화 값 (SAM 등에서 사용). null이면 0~1 정규화(pixel/255) 수행</param>
    /// <param name="std">Std 정규화 값</param>
    public static DenseTensor<float> ToTensor(this SKBitmap image, int width, int height, float[]? mean = null, float[]? std = null)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });

        // 리사이즈 또는 포맷 변환이 필요한 경우 처리
        // SkiaSharp에서 바이트 단위 접근 시 RGBA 순서를 보장하기 위해 SKColorType.Rgba8888로 강제 변환
        bool needsDispose = false;
        SKBitmap processedImage = image;

        if (image.Width != width || image.Height != height || image.ColorType != SKColorType.Rgba8888)
        {
            // Resize 메서드는 새로운 SKBitmap을 반환합니다.
            var info = new SKImageInfo(width, height, SKColorType.Rgba8888);
            processedImage = image.Resize(info, new SKSamplingOptions(SKCubicResampler.Mitchell));
            needsDispose = true; // 새로 생성된 비트맵은 나중에 해제해야 함
        }

        // 리사이즈 실패 시 (null 반환 가능성 대응)
        if (processedImage == null)
            throw new InvalidOperationException("Failed to resize image.");

        bool useMeanStd = mean != null && std != null;

        // 픽셀 데이터에 고속으로 접근
        ReadOnlySpan<byte> pixels = processedImage.GetPixelSpan();
        int bytesPerPixel = processedImage.BytesPerPixel; // Rgba8888이므로 4

        for (int y = 0; y < height; y++)
        {
            int rowOffset = y * processedImage.RowBytes;

            for (int x = 0; x < width; x++)
            {
                int pixelOffset = rowOffset + (x * bytesPerPixel);

                // Rgba8888 포맷이므로 순서는 R, G, B, A
                byte r = pixels[pixelOffset];
                byte g = pixels[pixelOffset + 1];
                byte b = pixels[pixelOffset + 2];

                if (useMeanStd)
                {
                    // (Value - Mean) / Std
                    tensor[0, 0, y, x] = (r - mean![0]) / std![0];
                    tensor[0, 1, y, x] = (g - mean![1]) / std![1];
                    tensor[0, 2, y, x] = (b - mean![2]) / std![2];
                }
                else
                {
                    // 0 ~ 1 Normalize
                    tensor[0, 0, y, x] = r / 255.0f;
                    tensor[0, 1, y, x] = g / 255.0f;
                    tensor[0, 2, y, x] = b / 255.0f;
                }
            }
        }

        // 임시 생성된 비트맵 정리
        if (needsDispose)
        {
            processedImage.Dispose();
        }

        return tensor;
    }

    /// <summary>
    /// 마스크 이미지를 [1, 1, H, W] Tensor로 변환 (LaMa용)
    /// </summary>
    public static DenseTensor<float> ToMaskTensor(this SKBitmap mask, int width, int height)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 1, height, width });

        bool needsDispose = false;
        SKBitmap processedMask = mask;

        // 마스크 리사이즈 (NearestNeighbor와 유사하게 처리하려면 FilterQuality.None 사용)
        if (mask.Width != width || mask.Height != height)
        {
            // 포맷은 처리 편의를 위해 기본값 사용 (보통 Platform native or Rgba8888)
            // 여기서는 단순 값 체크만 하므로 포맷 강제 변환은 선택사항이나, 안전을 위해 Rgba8888 권장
            var info = new SKImageInfo(width, height, SKColorType.Rgba8888);
            processedMask = mask.Resize(info, new SKSamplingOptions(SKFilterMode.Nearest));
            needsDispose = true;
        }

        if (processedMask == null)
            throw new InvalidOperationException("Failed to resize mask.");

        ReadOnlySpan<byte> pixels = processedMask.GetPixelSpan();
        int bytesPerPixel = processedMask.BytesPerPixel;

        for (int y = 0; y < height; y++)
        {
            int rowOffset = y * processedMask.RowBytes;

            for (int x = 0; x < width; x++)
            {
                int pixelOffset = rowOffset + (x * bytesPerPixel);

                // Rgba8888 기준 Red 채널 등을 확인 (Grayscale이면 r=g=b)
                byte val = pixels[pixelOffset];

                // 127보다 크면 1, 아니면 0
                tensor[0, 0, y, x] = val > 127 ? 1.0f : 0.0f;
            }
        }

        if (needsDispose)
        {
            processedMask.Dispose();
        }

        return tensor;
    }
}
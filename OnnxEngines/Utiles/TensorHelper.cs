using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace OnnxEngines.Utils;

public static class TensorHelper
{
    /// <summary>
    /// ImageSharp 이미지를 NCHW [1, 3, H, W] 형태의 Tensor로 변환합니다.
    /// </summary>
    /// <param name="mean">Mean 정규화 값 (SAM 등에서 사용). null이면 0~1 정규화(pixel/255) 수행</param>
    /// <param name="std">Std 정규화 값</param>
    public static DenseTensor<float> ToTensor(this Image<Rgba32> image, int width, int height, float[]? mean = null, float[]? std = null)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });

        // 리사이즈가 필요한 경우만 수행
        if (image.Width != width || image.Height != height)
        {
            image.Mutate(x => x.Resize(width, height));
        }

        bool useMeanStd = mean != null && std != null;

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
                {
                    var p = row[x];

                    if (useMeanStd)
                    {
                        // (Value - Mean) / Std
                        tensor[0, 0, y, x] = (p.R - mean![0]) / std![0];
                        tensor[0, 1, y, x] = (p.G - mean![1]) / std![1];
                        tensor[0, 2, y, x] = (p.B - mean![2]) / std![2];
                    }
                    else
                    {
                        // 0 ~ 1 Normalize
                        tensor[0, 0, y, x] = p.R / 255.0f;
                        tensor[0, 1, y, x] = p.G / 255.0f;
                        tensor[0, 2, y, x] = p.B / 255.0f;
                    }
                }
            }
        });

        return tensor;
    }

    /// <summary>
    /// Grayscale 마스크 이미지를 [1, 1, H, W] Tensor로 변환 (LaMa용)
    /// </summary>
    public static DenseTensor<float> ToMaskTensor(this Image<L8> mask, int width, int height)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 1, height, width });

        if (mask.Width != width || mask.Height != height)
            mask.Mutate(x => x.Resize(width, height, KnownResamplers.NearestNeighbor));

        mask.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
                {
                    // 127보다 크면 1, 아니면 0
                    tensor[0, 0, y, x] = row[x].PackedValue > 127 ? 1.0f : 0.0f;
                }
            }
        });

        return tensor;
    }
}
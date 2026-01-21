using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using OnnxEngines.Utils;

namespace OnnxEngines.Face;

public class FaceDetector : BaseOnnxEngine, IFaceDetector
{
    private const int InputWidth = 320;
    private const int InputHeight = 240;

    public FaceDetector(string modelPath, bool useGpu = false) : base(modelPath, useGpu) { }
    
    public List<SKRectI> DetectFaces(byte[] imageBytes, float confThreshold = 0.7f)
    {
        if (_session == null) throw new System.InvalidOperationException("Model not loaded.");

        using var image = SKBitmap.Decode(imageBytes).Copy(SKColorType.Rgba8888);
        int origW = image.Width;
        int origH = image.Height;

        using var resized = image.Resize(new SKImageInfo(InputWidth, InputHeight), new SKSamplingOptions(SKCubicResampler.Mitchell));
        var inputTensor = new DenseTensor<float>(new[] { 1, 3, InputHeight, InputWidth });

        ReadOnlySpan<byte> pixels = resized.GetPixelSpan();
        int bpp = resized.BytesPerPixel;

        for (int y = 0; y < InputHeight; y++)
        {
            int rowOff = y * resized.RowBytes;
            for (int x = 0; x < InputWidth; x++)
            {
                int idx = rowOff + (x * bpp);
                // -127 / 128 정규화
                inputTensor[0, 0, y, x] = (pixels[idx] - 127.0f) / 128.0f;
                inputTensor[0, 1, y, x] = (pixels[idx + 1] - 127.0f) / 128.0f;
                inputTensor[0, 2, y, x] = (pixels[idx + 2] - 127.0f) / 128.0f;
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(), inputTensor)
        };

        using var results = _session.Run(inputs);
        var confidences = results.First(x => x.Name == "scores").AsTensor<float>();
        var boxes = results.First(x => x.Name == "boxes").AsTensor<float>();

        var candidates = new List<(SKRectI Rect, float Score)>();
        int numAnchors = confidences.Dimensions[1];

        for (int i = 0; i < numAnchors; i++)
        {
            float score = confidences[0, i, 1];
            if (score > confThreshold)
            {
                float x = boxes[0, i, 0] * origW;
                float y = boxes[0, i, 1] * origH;
                float w = (boxes[0, i, 2] - boxes[0, i, 0]) * origW;
                float h = (boxes[0, i, 3] - boxes[0, i, 1]) * origH;

                candidates.Add((SKRectI.Create((int)x, (int)y, (int)w, (int)h), score));
            }
        }

        return NMS(candidates);
    }

    private List<SKRectI> NMS(List<(SKRectI Rect, float Score)> boxes, float iouThreshold = 0.3f)
    {
        var result = new List<SKRectI>();
        var sorted = boxes.OrderByDescending(x => x.Score).ToList();

        while (sorted.Count > 0)
        {
            var current = sorted[0];
            result.Add(current.Rect);
            sorted.RemoveAt(0);
            sorted.RemoveAll(other => CalculateIoU(current.Rect, other.Rect) > iouThreshold);
        }
        return result;
    }

    private float CalculateIoU(SKRectI r1, SKRectI r2)
    {
        var intersect = SKRectI.Intersect(r1, r2);
        if (intersect.IsEmpty) return 0f;
        float intersectionArea = intersect.Width * intersect.Height;
        float unionArea = (r1.Width * r1.Height) + (r2.Width * r2.Height) - intersectionArea;
        return intersectionArea / unionArea;
    }

    public byte[] ApplyBlur(byte[] imageBytes, List<SKRectI> faces, int blurSigma = 15)
    {
        using var image = SKBitmap.Decode(imageBytes); // 원본 포맷 유지
        using var canvas = new SKCanvas(image);

        foreach (var face in faces)
        {
            var roi = SKRectI.Intersect(face, new SKRectI(0, 0, image.Width, image.Height));
            if (roi.Width <= 1 || roi.Height <= 1) continue;

            int safeSigma = Math.Min(blurSigma, Math.Min(roi.Width, roi.Height) / 4);
            if (safeSigma < 1) safeSigma = 1;

            // Skia에서 부분 블러링:
            // 1. 해당 영역(ROI)을 잘라냅니다.
            using var subset = new SKBitmap(roi.Width, roi.Height);
            image.ExtractSubset(subset, roi);

            // 2. 잘라낸 이미지에 블러 효과를 적용하여 그릴 Paint 생성
            using var paint = new SKPaint();
            paint.ImageFilter = SKImageFilter.CreateBlur(safeSigma, safeSigma);

            // 3. 캔버스를 해당 ROI로 클립하고 블러된 이미지를 덮어씁니다.
            // (더 쉬운 방법: ROI 영역에 subset 이미지를 Blur 필터와 함께 그리기)
            canvas.Save();
            canvas.ClipRect(roi);
            canvas.DrawBitmap(subset, roi.Left, roi.Top, paint);
            canvas.Restore();
        }

        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        return data.ToArray();
    }

    public byte[] DrawBoundingBoxes(byte[] imageBytes, List<SKRectI> faces, float thickness = 3)
    {
        using var image = SKBitmap.Decode(imageBytes);
        using var canvas = new SKCanvas(image);
        using var paint = new SKPaint
        {
            Color = SKColors.Red,
            Style = SKPaintStyle.Stroke,
            StrokeWidth = thickness,
            IsAntialias = true
        };

        foreach (var face in faces)
        {
            canvas.DrawRect(face, paint);
        }

        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        return data.ToArray();
    }
}
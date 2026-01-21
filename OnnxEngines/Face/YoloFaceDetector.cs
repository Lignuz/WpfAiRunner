using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using OnnxEngines.Utils;

namespace OnnxEngines.Face;

public class YoloFaceDetector : BaseOnnxEngine, IFaceDetector
{
    // YOLOv8n-Face 입력 크기
    private const int InputSize = 640;

    public YoloFaceDetector(string modelPath, bool useGpu = false) : base(modelPath, useGpu) { }
    
    public List<SKRectI> DetectFaces(byte[] imageBytes, float confThreshold = 0.5f)
    {
        if (_session == null) throw new System.InvalidOperationException("Model not loaded.");

        using var image = SKBitmap.Decode(imageBytes).Copy(SKColorType.Rgba8888);
        int origW = image.Width;
        int origH = image.Height;

        // 1. 전처리 (Resize 640x640)
        using var resized = image.Resize(new SKImageInfo(InputSize, InputSize), new SKSamplingOptions(SKCubicResampler.Mitchell));
        var inputTensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });

        ReadOnlySpan<byte> pixels = resized.GetPixelSpan();
        int bpp = resized.BytesPerPixel;

        for (int y = 0; y < InputSize; y++)
        {
            int rowOff = y * resized.RowBytes;
            for (int x = 0; x < InputSize; x++)
            {
                int idx = rowOff + (x * bpp);
                // 0..1 Normalize
                inputTensor[0, 0, y, x] = pixels[idx] / 255.0f;
                inputTensor[0, 1, y, x] = pixels[idx + 1] / 255.0f;
                inputTensor[0, 2, y, x] = pixels[idx + 2] / 255.0f;
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("images", inputTensor)
        };

        // 2. 추론
        using var results = _session.Run(inputs);
        // Output: [1, 5, 8400] (cx, cy, w, h, score)
        var output = results.First().AsTensor<float>();

        var candidates = new List<(SKRectI Rect, float Score)>();
        int anchors = output.Dimensions[2];

        for (int i = 0; i < anchors; i++)
        {
            float score = output[0, 4, i]; // Score
            if (score > confThreshold)
            {
                float cx = output[0, 0, i];
                float cy = output[0, 1, i];
                float w = output[0, 2, i];
                float h = output[0, 3, i];

                // Scale 복원
                float x = (cx - w / 2) * (origW / (float)InputSize);
                float y = (cy - h / 2) * (origH / (float)InputSize);
                float width = w * (origW / (float)InputSize);
                float height = h * (origH / (float)InputSize);

                candidates.Add((SKRectI.Create((int)x, (int)y, (int)width, (int)height), score));
            }
        }

        return NMS(candidates);
    }

    private List<SKRectI> NMS(List<(SKRectI Rect, float Score)> boxes, float iouThreshold = 0.45f)
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

    // FaceDetector와 동일한 로직 사용
    public byte[] ApplyBlur(byte[] imageBytes, List<SKRectI> faces, int blurSigma = 15)
    {
        using var image = SKBitmap.Decode(imageBytes);
        using var canvas = new SKCanvas(image);

        foreach (var face in faces)
        {
            var roi = SKRectI.Intersect(face, new SKRectI(0, 0, image.Width, image.Height));
            if (roi.Width <= 1 || roi.Height <= 1) continue;

            int safeSigma = Math.Min(blurSigma, Math.Min(roi.Width, roi.Height) / 4);
            if (safeSigma < 1) safeSigma = 1;

            using var subset = new SKBitmap(roi.Width, roi.Height);
            image.ExtractSubset(subset, roi);

            using var paint = new SKPaint();
            paint.ImageFilter = SKImageFilter.CreateBlur(safeSigma, safeSigma);

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

        foreach (var face in faces) canvas.DrawRect(face, paint);

        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        return data.ToArray();
    }
}
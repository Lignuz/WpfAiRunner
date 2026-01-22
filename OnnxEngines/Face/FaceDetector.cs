using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace OnnxEngines.Face;

public class FaceDetector : BaseOnnxEngine, IFaceDetector
{
    private const int InputWidth = 320;
    private const int InputHeight = 240;

    public FaceDetector(string modelPath, bool useGpu = false) : base(modelPath, useGpu) { }

    public List<SKRectI> DetectFaces(byte[] imageBytes, float confThreshold = 0.7f)
    {
        if (_session == null) throw new System.InvalidOperationException("Model not loaded.");

        using var originalImage = SKBitmap.Decode(imageBytes);
        // 픽셀 포맷 고정 (알파 채널 이슈 방지)
        using var image = originalImage.Copy(SKColorType.Rgba8888);

        int origW = image.Width;
        int origH = image.Height;

        // --- Letterbox 리사이즈 (비율 유지) ---
        // 가로/세로 비율 중 더 많이 줄여야 하는 쪽을 기준으로 Scale 결정
        float ratio = Math.Min((float)InputWidth / origW, (float)InputHeight / origH);
        int newW = (int)(origW * ratio);
        int newH = (int)(origH * ratio);

        using var resized = image.Resize(new SKImageInfo(newW, newH), new SKSamplingOptions(SKCubicResampler.Mitchell));

        // 모델 입력 크기(320x240)의 빈 캔버스 생성 (기본값 검정 0,0,0)
        using var letterboxImage = new SKBitmap(InputWidth, InputHeight);
        using (var canvas = new SKCanvas(letterboxImage))
        {
            canvas.Clear(SKColors.Black);
            // 중앙 정렬을 위한 좌표 계산
            float padX = (InputWidth - newW) / 2f;
            float padY = (InputHeight - newH) / 2f;
            canvas.DrawBitmap(resized, padX, padY);
        }
        // ---------------------------------------------

        var inputTensor = new DenseTensor<float>(new[] { 1, 3, InputHeight, InputWidth });
        var pixels = letterboxImage.GetPixelSpan();
        int rowBytes = letterboxImage.RowBytes;

        // 정규화 루프 (기존 로직 유지: -1 ~ 1 Range)
        for (int y = 0; y < InputHeight; y++)
        {
            int rowOff = y * rowBytes;
            for (int x = 0; x < InputWidth; x++)
            {
                int idx = rowOff + (x * 4); // Rgba8888 = 4 bytes
                inputTensor[0, 0, y, x] = (pixels[idx] - 127.0f) / 128.0f;     // R
                inputTensor[0, 1, y, x] = (pixels[idx + 1] - 127.0f) / 128.0f; // G
                inputTensor[0, 2, y, x] = (pixels[idx + 2] - 127.0f) / 128.0f; // B
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

        // 좌표 복원을 위한 값들
        float padX_restore = (InputWidth - newW) / 2f;
        float padY_restore = (InputHeight - newH) / 2f;
        float scale = 1f / ratio;

        for (int i = 0; i < numAnchors; i++)
        {
            float score = confidences[0, i, 1]; // 1 = Face Class
            if (score > confThreshold)
            {
                // 모델 출력 (0.0 ~ 1.0 정규화된 좌표) -> 320x240 기준 픽셀 좌표로 변환
                float box_x1 = boxes[0, i, 0] * InputWidth;
                float box_y1 = boxes[0, i, 1] * InputHeight;
                float box_x2 = boxes[0, i, 2] * InputWidth;
                float box_y2 = boxes[0, i, 3] * InputHeight;

                // --- 좌표 원복 (Padding 제거 및 Scale 적용) ---
                // 1. Padding 제거
                float r_x1 = (box_x1 - padX_restore);
                float r_y1 = (box_y1 - padY_restore);
                float r_x2 = (box_x2 - padX_restore);
                float r_y2 = (box_y2 - padY_restore);

                // 2. 원본 해상도로 스케일링
                float x = r_x1 * scale;
                float y = r_y1 * scale;
                float w = (r_x2 - r_x1) * scale;
                float h = (r_y2 - r_y1) * scale;

                // 좌표 유효성 검사 (음수 방지 및 이미지 범위 제한)
                int finalX = Math.Max(0, (int)x);
                int finalY = Math.Max(0, (int)y);
                int finalW = Math.Min(origW - finalX, (int)w);
                int finalH = Math.Min(origH - finalY, (int)h);

                if (finalW > 0 && finalH > 0)
                {
                    candidates.Add((SKRectI.Create(finalX, finalY, finalW, finalH), score));
                }
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
        using var image = SKBitmap.Decode(imageBytes);
        using var canvas = new SKCanvas(image);
        using var paint = new SKPaint();

        foreach (var face in faces)
        {
            var roi = SKRectI.Intersect(face, new SKRectI(0, 0, image.Width, image.Height));
            if (roi.Width <= 1 || roi.Height <= 1) continue;

            int safeSigma = Math.Min(blurSigma, Math.Min(roi.Width, roi.Height) / 4);
            if (safeSigma < 1) safeSigma = 1;

            paint.ImageFilter = SKImageFilter.CreateBlur(safeSigma, safeSigma);

            // --- 자연스러운 블러 처리 ---
            // Subset을 만들지 않고 ClipRect를 사용해 원본 위에 블러를 덧그림
            // (경계선이 부드럽게 처리됨)
            canvas.Save();
            canvas.ClipRect(roi);
            canvas.DrawBitmap(image, 0, 0, paint); // 전체 이미지를 블러 필터로 그림 (Clip된 영역만 보임)
            canvas.Restore();

            paint.ImageFilter = null; // 필터 초기화
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
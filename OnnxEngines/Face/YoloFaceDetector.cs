using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace OnnxEngines.Face;

public class YoloFaceDetector : BaseOnnxEngine, IFaceDetector
{
    // YOLOv8n-Face 입력 크기
    private const int InputSize = 640;

    public YoloFaceDetector(string modelPath, bool useGpu = false) : base(modelPath, useGpu) { }

    public List<SKRectI> DetectFaces(byte[] imageBytes, float confThreshold = 0.5f)
    {
        if (_session == null) throw new System.InvalidOperationException("Model not loaded.");

        // 1. 디코드 시 알파 문제 방지를 위해 Opaque로 변환 시도 가능하지만, 
        // 보통 Copy(Rgba8888)로 픽셀 포맷은 맞췄으므로 비율 처리에 집중합니다.
        using var originalImage = SKBitmap.Decode(imageBytes);

        // Convert to standard RGBA (remove weird colortypes)
        using var image = originalImage.Copy(SKColorType.Rgba8888);

        int origW = image.Width;
        int origH = image.Height;

        // --- Letterbox 전처리 (비율 유지 + 여백 채우기) ---
        float ratio = Math.Min((float)InputSize / origW, (float)InputSize / origH);
        int newW = (int)(origW * ratio);
        int newH = (int)(origH * ratio);

        // 여백 계산 (중앙 정렬을 원하면 /2 하지만, YOLO는 보통 0,0 정렬 또는 중앙 정렬 학습에 따라 다름)
        // YOLOv8 기본: 중앙 정렬 권장, 혹은 단순 Rescale
        // 여기서는 단순하게 상단/좌측 정렬 후 남는 곳 블랙 처리 (구현이 가장 확실함)

        using var resized = image.Resize(new SKImageInfo(newW, newH), new SKSamplingOptions(SKCubicResampler.Mitchell));

        // 640x640 캔버스(검은색 배경) 생성
        using var letterboxImage = new SKBitmap(InputSize, InputSize);
        using (var canvas = new SKCanvas(letterboxImage))
        {
            canvas.Clear(SKColors.Black); // 배경 검은색 (Padding)
                                          // 중앙 정렬 (선택사항, 학습 모델에 따라 다를 수 있으나 보통 중앙이 안전)
            float padX = (InputSize - newW) / 2f;
            float padY = (InputSize - newH) / 2f;
            canvas.DrawBitmap(resized, padX, padY);
        }
        // -------------------------------------------------------

        var inputTensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });
        var pixels = letterboxImage.GetPixelSpan();

        // 픽셀 정규화 루프 
        for (int y = 0; y < InputSize; y++)
        {
            int rowOff = y * letterboxImage.RowBytes;
            for (int x = 0; x < InputSize; x++)
            {
                int idx = rowOff + (x * 4); // Rgba8888 = 4 bytes
                inputTensor[0, 0, y, x] = pixels[idx] / 255.0f;     // R
                inputTensor[0, 1, y, x] = pixels[idx + 1] / 255.0f; // G
                inputTensor[0, 2, y, x] = pixels[idx + 2] / 255.0f; // B
            }
        }

        // 추론 수행
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", inputTensor) };
        using var results = _session.Run(inputs);
        var output = results.First().AsTensor<float>();

        var candidates = new List<(SKRectI Rect, float Score)>();
        int anchors = output.Dimensions[2];

        // --- 좌표 복원 (Letterbox 고려) ---
        // Letterbox 적용 시의 Padding 값
        float padX_restore = (InputSize - newW) / 2f;
        float padY_restore = (InputSize - newH) / 2f;
        float scale = 1f / ratio; // 축소 비율의 역수

        for (int i = 0; i < anchors; i++)
        {
            float score = output[0, 4, i];
            if (score > confThreshold)
            {
                float cx = output[0, 0, i];
                float cy = output[0, 1, i];
                float w = output[0, 2, i];
                float h = output[0, 3, i];

                // 1. Padding 제거 (Letterbox 좌표 -> 리사이즈된 이미지 내 좌표)
                float x_cent = (cx - padX_restore);
                float y_cent = (cy - padY_restore);

                // 2. 원본 크기로 스케일링
                float x = (x_cent - w / 2) * scale;
                float y = (y_cent - h / 2) * scale;
                float width = w * scale;
                float height = h * scale;

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

        using var paint = new SKPaint();
        // 전체 이미지에 대한 블러 필터를 미리 생성 (성능상 큰 차이가 없다면 이 방식이 경계면 처리에 유리)
        // 혹은 ROI보다 약간 크게 잘라서 블러 후 ClipRect로 그림

        foreach (var face in faces)
        {
            var roi = SKRectI.Intersect(face, new SKRectI(0, 0, image.Width, image.Height));
            if (roi.Width <= 1 || roi.Height <= 1) continue;

            int safeSigma = Math.Min(blurSigma, Math.Min(roi.Width, roi.Height) / 4);
            if (safeSigma < 1) safeSigma = 1;

            // Subset을 만들지 않고, Canvas 레이어를 활용
            paint.ImageFilter = SKImageFilter.CreateBlur(safeSigma, safeSigma);

            canvas.Save();
            canvas.ClipRect(roi); // 얼굴 영역만 보이도록 클리핑
                                  // 이미지 전체(혹은 ROI 주변)를 다시 그려서, 블러 필터가 주변 픽셀과 섞이게 함
                                  // (단, 이 방식은 원본 위에 원본을 블러해서 덮어씌우는 방식)
            canvas.DrawBitmap(image, 0, 0, paint);
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

        foreach (var face in faces) canvas.DrawRect(face, paint);

        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        return data.ToArray();
    }
}
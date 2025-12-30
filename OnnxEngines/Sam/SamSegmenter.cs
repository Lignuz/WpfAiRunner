using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Collections.Concurrent;

namespace SamEngine;

// ISamSegmenter 인터페이스 구현
public class SamSegmenter : ISamSegmenter
{
    private InferenceSession? _encoderSession;
    private InferenceSession? _decoderSession;

    // SAM 모델의 표준 입력 해상도 (1024x1024)
    private const int TargetSize = 1024;

    // Encoder가 생성한 이미지 임베딩 데이터 (Decoder 입력용, 캐싱됨)
    private float[]? _imageEmbeddings;

    // 원본 이미지 크기
    private int _orgW, _orgH;

    // 전처리(리사이징)된 실제 이미지 크기 (패딩 제외)
    private int _resizedW, _resizedH;

    // 추론된 마스크 결과 텐서를 임시 보관하는 변수 (지연 생성을 위함)
    private Tensor<float>? _lastMaskTensor;

    public string DeviceMode { get; private set; } = "CPU";

    /// <summary>
    /// ONNX 모델(Encoder, Decoder)을 로드합니다.
    /// </summary>
    public void LoadModels(string encoderPath, string decoderPath, bool useGpu)
    {
        var so = new SessionOptions();

        if (useGpu)
        {
            try
            {
                so.AppendExecutionProvider_CUDA(0);
                DeviceMode = "GPU";
            }
            catch
            {
                DeviceMode = "CPU";
            }
        }

        _encoderSession = new InferenceSession(encoderPath, so);
        _decoderSession = new InferenceSession(decoderPath, so);
    }

    /// <summary>
    /// 이미지를 전처리(비율 유지 리사이징 + 패딩)하고 Encoder를 실행하여 임베딩을 생성합니다.
    /// </summary>
    public void EncodeImage(byte[] imageBytes)
    {
        if (_encoderSession == null)
            throw new InvalidOperationException("Encoder model is not loaded.");

        using var image = Image.Load<Rgba32>(imageBytes);
        _orgW = image.Width;
        _orgH = image.Height;

        // 긴 변을 1024로 맞추는 스케일 비율 계산
        float scale = (float)TargetSize / Math.Max(_orgW, _orgH);
        _resizedW = (int)(_orgW * scale);
        _resizedH = (int)(_orgH * scale);

        // 1. 비율 유지 리사이징
        image.Mutate(x => x.Resize(_resizedW, _resizedH));

        // 2. 1024x1024 검은색 캔버스 생성 (Padding 영역)
        using var paddedImage = new Image<Rgba32>(TargetSize, TargetSize);
        paddedImage.Mutate(x => x.BackgroundColor(Color.Black));

        // 3. 리사이징된 이미지를 좌상단(0,0)에 배치
        paddedImage.Mutate(x => x.DrawImage(image, new Point(0, 0), 1f));

        // Encoder 실행
        string inputName = _encoderSession.InputMetadata.Keys.First();
        var inputTensor = CreateEncoderInputTensor(paddedImage);

        using var results = _encoderSession.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        });

        var outputTensor = results.First().AsTensor<float>();
        _imageEmbeddings = ConvertEmbeddingsToCHW(outputTensor);
    }

    /// <summary>
    /// 좌표 프롬프트를 받아 Decoder를 실행합니다.
    /// 가장 점수가 높은 마스크 이미지는 즉시 생성하여 반환하고, 나머지는 텐서 형태로 캐싱합니다.
    /// </summary>
    public (List<float> Scores, byte[] BestMaskBytes, int BestIndex) Predict(float x, float y)
    {
        if (_decoderSession == null || _imageEmbeddings == null)
            return (new List<float>(), Array.Empty<byte>(), -1);

        // Embedding Tensor 준비
        var embedTensor = new DenseTensor<float>(_imageEmbeddings, new[] { 1, 256, 64, 64 });

        // 좌표 변환 (원본 좌표 -> 1024 모델 입력 좌표)
        var pointCoords = new DenseTensor<float>(new[] { 1, 2, 2 });
        float scale = (float)TargetSize / Math.Max(_orgW, _orgH);

        pointCoords[0, 0, 0] = x * scale;
        pointCoords[0, 0, 1] = y * scale;
        pointCoords[0, 1, 0] = 0f; // Padding Point
        pointCoords[0, 1, 1] = 0f;

        // 라벨 설정 (1=Positive Click, -1=Padding)
        var pointLabels = new DenseTensor<float>(new[] { 1, 2 });
        pointLabels[0, 0] = 1.0f;
        pointLabels[0, 1] = -1.0f;

        // 기타 필수 입력 텐서 준비
        var maskInput = new DenseTensor<float>(new[] { 1, 1, 256, 256 });
        var hasMaskInput = new DenseTensor<float>(new[] { 0.0f }, new[] { 1 });
        var origImSize = new DenseTensor<float>(new[] { (float)TargetSize, (float)TargetSize }, new[] { 2 });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image_embeddings", embedTensor),
            NamedOnnxValue.CreateFromTensor("point_coords", pointCoords),
            NamedOnnxValue.CreateFromTensor("point_labels", pointLabels),
            NamedOnnxValue.CreateFromTensor("mask_input", maskInput),
            NamedOnnxValue.CreateFromTensor("has_mask_input", hasMaskInput),
            NamedOnnxValue.CreateFromTensor("orig_im_size", origImSize),
        };

        // Decoder 추론 실행
        using var decResults = _decoderSession.Run(inputs);

        var maskResult = decResults.FirstOrDefault(r => r.Name == "masks") ?? decResults.First();
        var iouResult = decResults.FirstOrDefault(r => r.Name == "iou_predictions");

        // 추론된 마스크 텐서를 메모리에 복사하여 캐싱 (다른 후보 선택 시 사용)
        _lastMaskTensor = maskResult.AsTensor<float>().ToDenseTensor();

        var iouTensor = iouResult?.AsTensor<float>();
        var scores = new List<float>();

        // 점수 계산 및 최고 점수 인덱스 찾기
        int candidateCount = _lastMaskTensor.Dimensions[1];
        int bestIndex = 0;
        float maxScore = -1f;

        for (int i = 0; i < candidateCount; i++)
        {
            float rawScore = iouTensor != null ? iouTensor[0, i] : 0.0f;
            if (rawScore > 1.0f) rawScore = 1.0f;
            if (rawScore < 0.0f) rawScore = 0.0f;

            scores.Add(rawScore);

            if (rawScore > maxScore)
            {
                maxScore = rawScore;
                bestIndex = i;
            }
        }

        // [최적화] 가장 확실한 마스크 1장은 여기서 바로 생성 (UI 딜레이 제거)
        byte[] bestMaskBytes = MaskTensorToPng(_lastMaskTensor, bestIndex);

        return (scores, bestMaskBytes, bestIndex);
    }

    /// <summary>
    /// 캐싱된 마스크 텐서에서 특정 인덱스의 마스크만 이미지(PNG)로 변환하여 반환합니다.
    /// </summary>
    public byte[] GetMaskImage(int index)
    {
        if (_lastMaskTensor == null) return Array.Empty<byte>();
        return MaskTensorToPng(_lastMaskTensor, index);
    }

    // --- Private Helpers ---

    private DenseTensor<float> CreateEncoderInputTensor(Image<Rgba32> img1024)
    {
        const int H = 1024;
        const int W = 1024;
        const int C = 3;

        var tensor = new DenseTensor<float>(new[] { H, W, C });

        img1024.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < H; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < W; x++)
                {
                    var p = row[x];
                    tensor[y, x, 0] = (p.R - 123.675f) / 58.395f;
                    tensor[y, x, 1] = (p.G - 116.28f) / 57.12f;
                    tensor[y, x, 2] = (p.B - 103.53f) / 57.375f;
                }
            }
        });

        return tensor;
    }

    private float[] ConvertEmbeddingsToCHW(Tensor<float> outputTensor)
    {
        var dims = outputTensor.Dimensions.ToArray();
        var data = outputTensor.ToArray();
        var chw = new float[256 * 64 * 64];

        if (dims.Length == 4)
        {
            if (dims[1] == 256) // NCHW
            {
                Buffer.BlockCopy(data, 0, chw, 0, sizeof(float) * chw.Length);
                return chw;
            }
            if (dims[3] == 256) // NHWC -> NCHW 변환
            {
                int h = 64, w = 64, c = 256;
                Parallel.For(0, h * w, i =>
                {
                    int y = i / w;
                    int x = i % w;
                    int baseSrc = (y * w + x) * c;
                    int baseDst = y * w + x;
                    for (int ch = 0; ch < c; ch++)
                        chw[(ch * h * w) + baseDst] = data[baseSrc + ch];
                });
                return chw;
            }
        }

        if (chw.Length == data.Length)
            Buffer.BlockCopy(data, 0, chw, 0, sizeof(float) * chw.Length);

        return chw;
    }

    private byte[] MaskTensorToPng(Tensor<float> maskTensor, int maskIndex)
    {
        int rank = maskTensor.Dimensions.Length;
        int h = 256, w = 256;

        if (rank >= 2)
        {
            h = maskTensor.Dimensions[rank - 2];
            w = maskTensor.Dimensions[rank - 1];
        }

        using var rawMask = new Image<L8>(w, h);

        rawMask.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    float v = 0f;
                    if (rank == 4)
                        v = maskTensor[0, maskIndex, y, x];
                    else if (rank == 3)
                        v = maskTensor[maskIndex, y, x];
                    else
                        v = maskTensor[y, x];

                    row[x] = new L8(v > 0.0f ? (byte)255 : (byte)0);
                }
            }
        });

        // 원본 비율에 맞게 유효 영역 계산 (Dynamic Crop)
        double ratioW = (double)_resizedW / TargetSize;
        double ratioH = (double)_resizedH / TargetSize;

        int validW = (int)(w * ratioW);
        int validH = (int)(h * ratioH);

        validW = Math.Clamp(validW, 1, w);
        validH = Math.Clamp(validH, 1, h);

        rawMask.Mutate(x => x.Crop(new Rectangle(0, 0, validW, validH)));

        rawMask.Mutate(x => x.Resize(new ResizeOptions
        {
            Size = new Size(_orgW, _orgH),
            Mode = ResizeMode.Stretch
        }));

        using var ms = new MemoryStream();
        rawMask.SaveAsPng(ms);
        return ms.ToArray();
    }

    public void Dispose()
    {
        _encoderSession?.Dispose();
        _decoderSession?.Dispose();
    }
}
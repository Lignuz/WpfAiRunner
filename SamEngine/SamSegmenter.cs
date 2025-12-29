using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace SamEngine;

public class SamSegmenter : IDisposable
{
    private InferenceSession? _encoderSession;
    private InferenceSession? _decoderSession;

    // SAM 모델의 표준 입력 크기
    private const int TargetSize = 1024;

    // 인코더가 생성한 이미지 임베딩 데이터 (Decoder 입력용)
    private float[]? _imageEmbeddings;

    // 원본 이미지 크기
    private int _orgW, _orgH;

    // 리사이징된 실제 이미지 크기 (패딩 제외)
    private int _resizedW, _resizedH;

    public string DeviceMode { get; private set; } = "CPU";

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
    /// 이미지를 1024x1024 크기로 전처리(비율 유지 리사이징 + 패딩)하고 인코딩을 수행합니다.
    /// </summary>
    public void EncodeImage(byte[] imageBytes)
    {
        if (_encoderSession == null)
            throw new InvalidOperationException("Encoder model is not loaded.");

        using var image = Image.Load<Rgba32>(imageBytes);
        _orgW = image.Width;
        _orgH = image.Height;

        // 긴 변을 1024로 맞추는 스케일 계산
        float scale = (float)TargetSize / Math.Max(_orgW, _orgH);
        _resizedW = (int)(_orgW * scale);
        _resizedH = (int)(_orgH * scale);

        // 1. 비율 유지 리사이징
        image.Mutate(x => x.Resize(_resizedW, _resizedH));

        // 2. 1024x1024 검은색 캔버스 생성 (Padding)
        using var paddedImage = new Image<Rgba32>(TargetSize, TargetSize);
        paddedImage.Mutate(x => x.BackgroundColor(Color.Black));

        // 3. 리사이징된 이미지를 좌상단(0,0)에 배치
        paddedImage.Mutate(x => x.DrawImage(image, new Point(0, 0), 1f));

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
    /// 원본 이미지 좌표(x, y)를 받아 마스크를 생성합니다.
    /// </summary>
    public byte[] PredictMask(float x, float y)
    {
        if (_decoderSession == null)
            throw new InvalidOperationException("Decoder model is not loaded.");
        if (_imageEmbeddings == null)
            return Array.Empty<byte>();

        // Embedding Tensor 준비
        var embedTensor = new DenseTensor<float>(_imageEmbeddings, new[] { 1, 256, 64, 64 });

        // 좌표 변환: 원본 좌표 -> 1024 스케일 좌표
        float scale = (float)TargetSize / Math.Max(_orgW, _orgH);

        var pointCoords = new DenseTensor<float>(new[] { 1, 2, 2 });
        pointCoords[0, 0, 0] = x * scale;
        pointCoords[0, 0, 1] = y * scale;
        pointCoords[0, 1, 0] = 0f; // Padding Point
        pointCoords[0, 1, 1] = 0f;

        // 라벨 설정 (1=포함, -1=패딩)
        var pointLabels = new DenseTensor<float>(new[] { 1, 2 });
        pointLabels[0, 0] = 1.0f;
        pointLabels[0, 1] = -1.0f;

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

        using var decResults = _decoderSession.Run(inputs);
        var maskTensor = decResults.First().AsTensor<float>();

        return MaskTensorToPng(maskTensor);
    }

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
                    // SAM 표준 정규화
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

        // 출력 형태에 따라 데이터 레이아웃 재배치 (N, H, W, C -> N, C, H, W 등)
        if (dims.Length == 4)
        {
            if (dims[1] == 256) // 이미 CHW 형태
            {
                Buffer.BlockCopy(data, 0, chw, 0, sizeof(float) * chw.Length);
                return chw;
            }
            if (dims[3] == 256) // NHWC 형태 -> CHW로 변환
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

        // 예외 처리: 차원 정보가 3개인 경우 등은 상황에 맞게 추가 구현 필요
        // 현재는 일반적인 [1, 256, 64, 64]와 [1, 64, 64, 256]만 처리
        if (chw.Length != data.Length)
            Buffer.BlockCopy(data, 0, chw, 0, sizeof(float) * chw.Length); // 단순 복사 시도
        else
            return data; // fallback

        return chw;
    }

    private byte[] MaskTensorToPng(Tensor<float> maskTensor)
    {
        int rank = maskTensor.Dimensions.Length;
        int h = 256, w = 256;

        // 텐서의 실제 차원 크기 확인
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
                    if (rank == 4) v = maskTensor[0, 0, y, x];
                    else if (rank == 3) v = maskTensor[0, y, x];
                    else if (rank == 2) v = maskTensor[y, x];

                    // Logit 값이 0보다 크면 마스크 영역
                    row[x] = new L8(v > 0.0f ? (byte)255 : (byte)0);
                }
            }
        });

        // 유효 영역 계산: 입력(1024) 대비 실제 이미지(_resizedW)가 차지하는 비율 계산
        // 이를 통해 모델 출력이 256이든 1024든 상관없이 정확한 비율로 크롭 가능
        double ratioW = (double)_resizedW / TargetSize;
        double ratioH = (double)_resizedH / TargetSize;

        int validW = (int)(w * ratioW);
        int validH = (int)(h * ratioH);

        validW = Math.Clamp(validW, 1, w);
        validH = Math.Clamp(validH, 1, h);

        // 검은색 패딩 영역 제거 (Crop)
        rawMask.Mutate(x => x.Crop(new Rectangle(0, 0, validW, validH)));

        // 원본 크기로 복원 (Stretch)
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
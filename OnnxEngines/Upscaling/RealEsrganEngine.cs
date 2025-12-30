using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace EsrganEngine;

public class RealEsrganEngine : IDisposable
{
    private InferenceSession? _session;
    public string DeviceMode { get; private set; } = "CPU";

    // [설정]
    // ModelInputSize: 모델이 요구하는 입력 크기 (고정)
    // Overlap: 타일 간 겹치는 영역 크기 (가장자리 아티팩트 제거용)
    // StepSize: 실제 유효하게 처리되는 타일의 이동 간격 (Input - 2*Overlap)

    private const int ModelInputSize = 128;
    private const int Overlap = 14; // 가장자리 14px 씩은 버림 (충분한 컨텍스트 확보)
    private const int StepSize = ModelInputSize - (Overlap * 2); // 128 - 28 = 100px 씩 이동

    public void LoadModel(string modelPath, bool useGpu)
    {
        _session?.Dispose();

        var so = new SessionOptions();
        if (useGpu)
        {
            try
            {
                so.AppendExecutionProvider_CUDA(0);
                DeviceMode = "GPU";
            }
            catch { DeviceMode = "CPU"; }
        }
        else
        {
            DeviceMode = "CPU";
        }

        _session = new InferenceSession(modelPath, so);

        // GPU 웜업
        if (DeviceMode == "GPU")
        {
            try
            {
                var dummyTensor = new DenseTensor<float>(new[] { 1, 3, ModelInputSize, ModelInputSize });
                string inputName = _session.InputMetadata.Keys.First();
                using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, dummyTensor) });
            }
            catch { }
        }
    }

    public byte[] Upscale(byte[] imageBytes, IProgress<double>? progress = null)
    {
        if (_session == null) throw new InvalidOperationException("Model not loaded.");

        using var srcImage = Image.Load<Rgba32>(imageBytes);
        int w = srcImage.Width;
        int h = srcImage.Height;

        // 1. 전체 이미지 패딩 (오버랩 처리를 위해 상하좌우 확장)
        // 가장자리를 복사(Duplicate)해서 채우는 것이 이상적이나, 간단히 배경색으로 확장 후 처리
        int padW = w + (Overlap * 2);
        int padH = h + (Overlap * 2);

        using var paddedSrc = new Image<Rgba32>(padW, padH);

        // 원본을 (Overlap, Overlap) 위치에 그리기 (중앙 배치)
        paddedSrc.Mutate(x => x.DrawImage(srcImage, new Point(Overlap, Overlap), 1f));

        // 가장자리 픽셀 복사 (Edge Replication) - 검은 테두리 방지
        // (ImageSharp에는 간단한 Edge Replication 기능이 없어, 여기서는 생략하고 Overlap 영역을 검은색으로 둡니다.
        //  Overlap이 충분하면 실제 그림 영역 안쪽의 경계선은 사라집니다.)

        // 2. 결과 이미지 준비 (4배 크기)
        int outW = w * 4;
        int outH = h * 4;
        using var resultImage = new Image<Rgba32>(outW, outH);

        // 3. 타일링 루프 설정
        // 패딩된 이미지 기준으로 (0,0)부터 StepSize(100)만큼 이동하며 128x128을 뜯어냄
        // 실제 유효 데이터는 128의 중앙 100 부분임.

        // X축 타일 개수 계산
        int countX = (int)Math.Ceiling((double)w / StepSize);
        int countY = (int)Math.Ceiling((double)h / StepSize);
        int totalTiles = countX * countY;
        int processedCount = 0;

        for (int y = 0; y < countY; y++)
        {
            for (int x = 0; x < countX; x++)
            {
                // 입력 타일 좌표 (패딩된 이미지 기준)
                int srcX = x * StepSize;
                int srcY = y * StepSize;

                // 마지막 타일이 이미지 범위를 넘지 않도록 조정
                if (srcX + ModelInputSize > padW) srcX = padW - ModelInputSize;
                if (srcY + ModelInputSize > padH) srcY = padH - ModelInputSize;

                // 타일 잘라내기 (128x128)
                using var tile = paddedSrc.Clone(ctx => ctx.Crop(new Rectangle(srcX, srcY, ModelInputSize, ModelInputSize)));

                // 모델 실행 (128 -> 512)
                using var upscaledTile = ProcessTile(tile);

                // 결과 붙여넣기 로직
                // 모델 출력(512) 중에서 유효한 중앙 부분(StepSize * 4 = 400)만 잘라서 붙임
                // 단, 맨 가장자리 타일은 오버랩을 포함해서 붙여야 빈틈이 안 생김.

                // 붙여넣을 위치 (원본 해상도 4배 기준)
                int destX = x * StepSize * 4;
                int destY = y * StepSize * 4;

                // 잘라낼 영역 (Output 타일 기준)
                int cropX = Overlap * 4;
                int cropY = Overlap * 4;
                int cropW = StepSize * 4;
                int cropH = StepSize * 4;

                // 경계 조건 처리: 마지막 타일 등 위치 보정
                // (단순화: 여기서는 StepSize 만큼씩 정확히 붙여넣음. 
                //  마지막 타일의 경우 약간 겹쳐서 덮어쓰게 되는데 이는 자연스러운 블렌딩 효과를 줌)

                // 실제 출력 위치 보정 (마지막 타일인 경우)
                if (destX + cropW > outW) destX = outW - cropW;
                if (destY + cropH > outH) destY = outH - cropH;

                // 유효 영역 크롭 및 그리기
                // (upscaledTile: 512x512 -> 유효영역 400x400 만 잘라서 dest에 그리기)
                // *주의: 모델 출력 전체를 덮어쓰면 안되고, 가장자리를 날려야 함.

                // 이번에는 조금 더 단순하고 강력한 방법: 
                // "Overlap을 제외한 중앙부"만 잘라서 1:1로 매핑

                var validRect = new Rectangle(cropX, cropY, cropW, cropH);

                // 마지막 타일 등 크기 예외처리
                // (마지막 타일은 오른쪽/아래 끝까지 채워야 하므로, ValidRect를 조금 더 넓게 잡거나 위치를 조정)
                // 위에서 srcX를 조정했으므로, 결과물은 항상 꽉 차게 나옴.

                using var validPart = upscaledTile.Clone(ctx => ctx.Crop(validRect));
                resultImage.Mutate(ctx => ctx.DrawImage(validPart, new Point(destX, destY), 1f));

                // 진행률
                processedCount++;
                progress?.Report((double)processedCount / totalTiles);
            }
        }

        using var ms = new MemoryStream();
        resultImage.SaveAsPng(ms);
        return ms.ToArray();
    }

    private Image<Rgba32> ProcessTile(Image<Rgba32> tileImage)
    {
        // 전처리 (0~1 float)
        var inputTensor = new DenseTensor<float>(new[] { 1, 3, ModelInputSize, ModelInputSize });
        tileImage.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < ModelInputSize; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < ModelInputSize; x++)
                {
                    inputTensor[0, 0, y, x] = row[x].R / 255.0f;
                    inputTensor[0, 1, y, x] = row[x].G / 255.0f;
                    inputTensor[0, 2, y, x] = row[x].B / 255.0f;
                }
            }
        });

        // 추론
        var inputName = _session!.InputMetadata.Keys.First();
        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) });
        var outputTensor = results.First().AsTensor<float>();

        // 후처리 (0~255 byte)
        int outSize = ModelInputSize * 4; // 512
        var outputImage = new Image<Rgba32>(outSize, outSize);
        outputImage.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < outSize; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < outSize; x++)
                {
                    float r = Math.Clamp(outputTensor[0, 0, y, x], 0, 1) * 255;
                    float g = Math.Clamp(outputTensor[0, 1, y, x], 0, 1) * 255;
                    float b = Math.Clamp(outputTensor[0, 2, y, x], 0, 1) * 255;
                    row[x] = new Rgba32((byte)r, (byte)g, (byte)b);
                }
            }
        });

        return outputImage;
    }

    public void Dispose() => _session?.Dispose();
}

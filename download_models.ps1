# PowerShell 5.1 호환성을 위해 어셈블리 로드 및 보안 프로토콜 설정
Add-Type -AssemblyName System.Net.Http
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12

# 화면 초기화 및 설정
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$baseDir = Get-Location
$modelsDir = Join-Path $baseDir "models"

Clear-Host
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   WpfAiRunner Model Downloader " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Target Directory: $modelsDir" -ForegroundColor Gray
Write-Host ""

# models 폴더 생성
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
    Write-Host "[Info] Created 'models' directory." -ForegroundColor Yellow
}

# 다운로드 함수 (이어받기/진행률 표시)
function Download-FileWithProgress {
    param (
        [string]$Url,
        [string]$OutputPath
    )

    try {
        $httpClient = New-Object System.Net.Http.HttpClient
        $httpClient.Timeout = [TimeSpan]::FromMinutes(30)

        # 헤더만 먼저 읽어서 파일 크기 확인
        $response = $httpClient.GetAsync($Url, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
        $response.EnsureSuccessStatusCode() | Out-Null

        $totalBytes = $response.Content.Headers.ContentLength
        # ContentLength가 없는 경우 대비
        if ($null -eq $totalBytes) { $totalBytes = 0 }
        $totalMB = $totalBytes / 1MB

        $contentStream = $response.Content.ReadAsStreamAsync().Result
        $fileStream = [System.IO.File]::Create($OutputPath)
        
        $buffer = New-Object byte[] 8192 # 8KB 버퍼
        $totalRead = 0
        $read = 0

        # 커서 숨기기 (깔끔한 출력을 위해)
        try { [Console]::CursorVisible = $false } catch {}

        do {
            $read = $contentStream.Read($buffer, 0, $buffer.Length)
            if ($read -gt 0) {
                $fileStream.Write($buffer, 0, $read)
                $totalRead += $read
                
                # 진행률 계산 및 출력
                $currentMB = $totalRead / 1MB
                Write-Host -NoNewline "`r    Progress: "
                if ($totalBytes -gt 0) {
                    $percent = ($totalRead / $totalBytes) * 100
                    Write-Host -NoNewline "$("{0:N2}" -f $currentMB) MB / $(" {0:N2}" -f $totalMB) MB " -ForegroundColor Yellow
                    Write-Host -NoNewline "($("{0:N1}" -f $percent)%)" -ForegroundColor Cyan
                } else {
                    # 전체 크기를 모를 경우 받은 양만 표시
                    Write-Host -NoNewline "$("{0:N2}" -f $currentMB) MB downloaded..." -ForegroundColor Yellow
                }
            }
        } while ($read -gt 0)
        Write-Host ""
    }
    catch {
        Write-Host "`n[Error] Download failed: $_" -ForegroundColor Red
        # 부분 파일 삭제
        if ($fileStream) { $fileStream.Close(); $fileStream.Dispose() }
        if (Test-Path $OutputPath) { Remove-Item $OutputPath }
        throw $_
    }
    finally {
        if ($fileStream) { $fileStream.Close(); $fileStream.Dispose() }
        if ($contentStream) { $contentStream.Close(); $contentStream.Dispose() }
        if ($httpClient) { $httpClient.Dispose() }
        try { [Console]::CursorVisible = $true } catch {}
    }
}

# 다운로드할 모델 목록
$models = @(
    @{ Name = "lama_fp32.onnx";                   Url = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx?download=true" },
    @{ Name = "depth_anything_v2_small.onnx";     Url = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx?download=true" },
    @{ Name = "mobile_sam.encoder.onnx";          Url = "https://huggingface.co/Acly/MobileSAM/resolve/main/mobile_sam_image_encoder.onnx?download=true" },
    @{ Name = "mobile_sam.decoder.onnx";          Url = "https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_multi.onnx?download=true" },
    @{ Name = "sam2_hiera_small.encoder.onnx";    Url = "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main/sam2_hiera_small.encoder.onnx?download=true" },
    @{ Name = "sam2_hiera_small.decoder.onnx";    Url = "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main/sam2_hiera_small.decoder.onnx?download=true" },
    @{ Name = "Real-ESRGAN-x4plus.onnx";          Url = "https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/resolve/01179a4da7bf5ac91faca650e6afbf282ac93933/Real-ESRGAN-x4plus.onnx" },
    @{ Name = "rmbg-1.4.onnx";                    Url = "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx?download=true" },
	@{ Name = "version-RFB-320.onnx";             Url = "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx?download=true" },
	@{ Name = "yolov8n-face.onnx";                Url = "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.onnx?download=true" },
	@{ Name = "yolov11n-face.onnx";               Url = "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov11n-face/model.onnx?download=true" },
	@{ Name = "AnimeGANv2_Hayao.onnx";    Url = "https://huggingface.co/vumichien/AnimeGANv2_Hayao/resolve/main/AnimeGANv2_Hayao.onnx?download=true" },
    @{ Name = "AnimeGANv2_Shinkai.onnx";  Url = "https://huggingface.co/vumichien/AnimeGANv2_Shinkai/resolve/main/AnimeGANv2_Shinkai.onnx?download=true" },
    @{ Name = "AnimeGANv2_Paprika.onnx";  Url = "https://huggingface.co/vumichien/AnimeGANv2_Paprika/resolve/main/AnimeGANv2_Paprika.onnx?download=true" },
	@{ Name = "ddcolor.onnx";             Url = "https://huggingface.co/crj/dl-ws/resolve/main/ddcolor.onnx?download=true" },
	@{ Name = "stable-diffusion-v1-5-Q8_0.gguf";             Url = "https://huggingface.co/gpustack/stable-diffusion-v1-5-GGUF/resolve/main/stable-diffusion-v1-5-Q8_0.gguf" }
)

# 메인 루프
foreach ($model in $models) {
    $outputPath = Join-Path $modelsDir $model.Name

    # 파일이 존재하면 다운로드 건너뜀 (Skip Logic)
    if (Test-Path $outputPath) {
        Write-Host "[Skip] $($model.Name)" -NoNewline -ForegroundColor DarkGray
        Write-Host " (Already exists)" -ForegroundColor Gray
    }
    else {
        Write-Host "[Down] Downloading $($model.Name)..." -ForegroundColor White
        try {
            Download-FileWithProgress -Url $model.Url -OutputPath $outputPath
            Write-Host "       Successfully downloaded." -ForegroundColor Green
        }
        catch {
             Write-Host "       Skipping..." -ForegroundColor DarkGray
        }
    }
    Write-Host ""
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   All downloads finished." -ForegroundColor Cyan
Read-Host -Prompt "Press Enter to exit"
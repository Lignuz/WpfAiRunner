using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using SkiaSharp;
using Sdcb.PaddleOCR;
using Sdcb.PaddleOCR.Models;
using Sdcb.PaddleOCR.Models.Local;
using OpenCvSharp;

namespace WpfAiRunner.Views;

public partial class OcrView : BaseAiView
{
    private PaddleOcrAll? _ocr;
    private PaddleOcrResult? _lastResult;
    private PaddleOcrResultRegion? _selectedRegion; // 클릭된 영역 저장
    private byte[]? _processedBytes;

    protected override Image ControlImgInput => ImgInput;
    protected override Image? ControlImgOutput => ImgOutput;
    protected override ProgressBar? ControlPbarLoading => PbarLoading;
    protected override TextBlock? ControlTxtStatus => TxtStatus;

    public OcrView() => InitializeComponent();

    public override void Dispose() => _ocr?.Dispose();

    protected override async void OnLoaded(RoutedEventArgs e) => await ReloadModel();

    protected override void OnImageLoaded()
    {
        _lastResult = null;
        _selectedRegion = null;
        TxtResult.Clear();
        ImgOutput.Source = null;
        UpdateUi();
    }

    private async void CboLangSelect_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!IsLoaded) return;
        await ReloadModel();
    }

    private async Task ReloadModel()
    {
        this.SetBusyState(true);
        try
        {
            _ocr?.Dispose();
            int langIdx = CboLangSelect.SelectedIndex;

            await Task.Run(() =>
            {
                // Local 패키지의 V4 모델 참조
                FullOcrModel model = langIdx == 0 ? LocalFullModels.KoreanV4 : LocalFullModels.EnglishV4;

                _ocr = new PaddleOcrAll(model)
                {
                    AllowRotateDetection = true,
                    Enable180Classification = true // v3 기준 속성명
                };
            });
            Log($"{((ComboBoxItem)CboLangSelect.SelectedItem).Content} OCR Loaded");
        }
        catch (Exception ex)
        {
            MessageBox.Show($"OCR Load Error: {ex.Message}");
        }
        finally
        {
            this.SetBusyState(false);
            UpdateUi();
        }
    }

    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        if (_ocr == null || _inputBitmap == null) return;

        this.SetBusyState(true);
        Log("OCR Running...");
        try
        {
            byte[] inputBytes = BitmapToBytes(_inputBitmap);

            _lastResult = await Task.Run(() =>
            {
                // OpenCvSharp을 이용한 디코드 및 실행
                using var mat = Cv2.ImDecode(inputBytes, ImreadModes.Color);
                return _ocr.Run(mat);
            });

            _selectedRegion = null; // 실행 시 선택 초기화
            TxtResult.Text = _lastResult.Text;
            await RenderResult();
        }
        catch (Exception ex)
        {
            Log($"OCR Error: {ex.Message}");
        }
        finally
        {
            this.SetBusyState(false);
        }
    }

    // 이미지 클릭 시 박스 감지 로직
    private async void ImgOutput_MouseDown(object sender, MouseButtonEventArgs e)
    {
        if (_lastResult == null || _inputBitmap == null || ImgOutput.Source == null) return;

        // 1. 클릭 좌표를 실제 이미지 좌표로 변환 (Stretch="Uniform" 대응)
        var clickPos = e.GetPosition(ImgOutput);

        double actualImgW = _inputBitmap.Width;
        double actualImgH = _inputBitmap.Height;
        double viewW = ImgOutput.ActualWidth;
        double viewH = ImgOutput.ActualHeight;

        double scale = Math.Min(viewW / actualImgW, viewH / actualImgH);
        double offsetX = (viewW - (actualImgW * scale)) / 2;
        double offsetY = (viewH - (actualImgH * scale)) / 2;

        double imgX = (clickPos.X - offsetX) / scale;
        double imgY = (clickPos.Y - offsetY) / scale;

        // 2. 클릭 지점이 포함된 영역 찾기
        var clickedRegion = _lastResult.Regions
            .Cast<PaddleOcrResultRegion?>() // Nullable로 캐스팅하여 검색
            .FirstOrDefault(r =>
            {
                if (r is not PaddleOcrResultRegion regionValue) return false;

                var pts = r.Value.Rect.Points();
                float minX = pts.Min(p => p.X);
                float maxX = pts.Max(p => p.X);
                float minY = pts.Min(p => p.Y);
                float maxY = pts.Max(p => p.Y);
                return imgX >= minX && imgX <= maxX && imgY >= minY && imgY <= maxY;
            });

        if (clickedRegion.HasValue)
        {
            _selectedRegion = clickedRegion;
            var region = clickedRegion.Value;
            TxtResult.Text = $"[Selected Region]\nText: {region.Text}\nScore: {region.Score:P1}";
            await RenderResult(); // 다시 그려서 하이라이트 표시
        }
    }

    private async Task RenderResult()
    {
        if (_lastResult == null || _inputBitmap == null) return;

        bool showBox = ChkShowBox.IsChecked == true;
        byte[] inputBytes = BitmapToBytes(_inputBitmap);

        await Task.Run(() =>
        {
            using var skiaBitmap = SKBitmap.Decode(inputBytes);
            using var surface = SKSurface.Create(new SKImageInfo(skiaBitmap.Width, skiaBitmap.Height));
            var canvas = surface.Canvas;
            canvas.DrawBitmap(skiaBitmap, 0, 0);

            if (showBox)
            {
                using var paintNormal = new SKPaint { Color = SKColors.Cyan, Style = SKPaintStyle.Stroke, StrokeWidth = 2 };
                using var paintSelected = new SKPaint { Color = SKColors.Yellow, Style = SKPaintStyle.Stroke, StrokeWidth = 4 };

                foreach (var region in _lastResult.Regions)
                {
                    var paint = (region == _selectedRegion) ? paintSelected : paintNormal;
                    var p = region.Rect.Points(); // Points() 메서드 호출

                    canvas.DrawLine((float)p[0].X, (float)p[0].Y, (float)p[1].X, (float)p[1].Y, paint);
                    canvas.DrawLine((float)p[1].X, (float)p[1].Y, (float)p[2].X, (float)p[2].Y, paint);
                    canvas.DrawLine((float)p[2].X, (float)p[2].Y, (float)p[3].X, (float)p[3].Y, paint);
                    canvas.DrawLine((float)p[3].X, (float)p[3].Y, (float)p[0].X, (float)p[0].Y, paint);
                }
            }

            using var img = surface.Snapshot();
            using var data = img.Encode(SKEncodedImageFormat.Png, 100);
            _processedBytes = data.ToArray();
        });

        if (_processedBytes != null)
        {
            ImgOutput.Source = BytesToBitmap(_processedBytes);
        }

        Log($"Rendered. {_lastResult.Regions.Length} regions.");
    }

    private void BtnCopy_Click(object sender, RoutedEventArgs e)
    {
        if (!string.IsNullOrEmpty(TxtResult.Text))
            Clipboard.SetText(TxtResult.Text);
    }

    private void BtnOpenImage_Click(object sender, RoutedEventArgs e) => OpenImageDialog();
    private void BtnSave_Click(object sender, RoutedEventArgs e) => SaveOutputImage();
    private async void ChkOption_Click(object sender, RoutedEventArgs e) => await RenderResult();

    private void UpdateUi()
    {
        bool busy = ControlPbarLoading?.Visibility == Visibility.Visible;
        BtnRun.IsEnabled = !busy && _ocr != null && _inputBitmap != null;
        BtnSave.IsEnabled = !busy && ImgOutput.Source != null;
    }
}
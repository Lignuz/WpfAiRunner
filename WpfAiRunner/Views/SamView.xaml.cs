using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using Microsoft.Win32;
using SamEngine;
using Path = System.IO.Path;

namespace WpfAiRunner.Views;

public partial class SamView : UserControl, IDisposable
{
    // 인터페이스 타입으로 선언 (동적 할당)
    private ISamSegmenter _segmenter;
    private BitmapSource? _inputBitmap;   // DPI 정규화된 입력 이미지
    private bool _isModelLoaded;
    private bool _isImageEncoded;
    private Point? _lastClickRatio;       // 오버레이 표시용 상대 좌표

    // 콤보박스 업데이트 중인지 체크하는 플래그
    private bool _isUpdatingCombo;

    public SamView()
    {
        InitializeComponent();
        // 기본값은 SAM 2로 설정 (초기화)
        _segmenter = new Sam2Segmenter();
    }

    public void Dispose() => _segmenter?.Dispose();

    private void UserControl_Loaded(object sender, RoutedEventArgs e) { }

    private async void BtnLoadModels_Click(object sender, RoutedEventArgs e)
    {
        // 1. 현재 선택된 모델 타입 확인
        int modelTypeIndex = CboModelType.SelectedIndex; // 0: MobileSAM, 1: SAM 2

        // 2. 엔진 교체
        _segmenter.Dispose();
        if (modelTypeIndex == 0) _segmenter = new SamSegmenter();
        else _segmenter = new Sam2Segmenter();

        var dlg = new OpenFileDialog { Title = $"Select {CboModelType.Text} Encoder", Filter = "ONNX|*.onnx" };
        if (dlg.ShowDialog(Window.GetWindow(this)) != true) return;

        string encoderPath = dlg.FileName;
        string folder = Path.GetDirectoryName(encoderPath)!;
        string encoderNameLower = Path.GetFileName(encoderPath).ToLower();

        // 3. 디코더 자동 찾기 로직 (엄격한 구분 적용)
        string? decoderPath = null;
        var allDecoders = Directory.GetFiles(folder, "*decoder*.onnx");

        if (modelTypeIndex == 1) // [SAM 2 모드]
        {
            // (1) 같은 변종(tiny, small 등)을 가진 SAM 2 디코더 우선 검색
            string[] variants = { "tiny", "small", "base_plus", "large" };
            string? detectedVariant = variants.FirstOrDefault(v => encoderNameLower.Contains(v));

            if (!string.IsNullOrEmpty(detectedVariant))
            {
                decoderPath = allDecoders.FirstOrDefault(f =>
                    Path.GetFileName(f).ToLower().Contains(detectedVariant) &&
                    Path.GetFileName(f).ToLower().Contains("sam2")); // sam2 키워드 필수
            }

            // (2) 없으면 'sam2'나 'hiera'가 들어간 아무 디코더 검색
            decoderPath ??= allDecoders.FirstOrDefault(f =>
            {
                string name = Path.GetFileName(f).ToLower();
                return name.Contains("sam2") || name.Contains("hiera");
            });
        }
        else // [MobileSAM 모드]
        {
            // (1) 'mobile_sam'이 들어간 디코더 우선 검색
            decoderPath = allDecoders.FirstOrDefault(f => Path.GetFileName(f).ToLower().Contains("mobile"));

            // (2) 없으면 일반 디코더를 찾되, **SAM 2용 파일은 제외**
            decoderPath ??= allDecoders.FirstOrDefault(f =>
            {
                string name = Path.GetFileName(f).ToLower();
                // [핵심] SAM 2 전용 키워드가 없는 파일만 선택
                return !name.Contains("sam2") && !name.Contains("hiera");
            });
        }

        // 4. 그래도 못 찾았으면 인코더 경로를 임시로 넣어서 아래 확인창 띄움
        decoderPath ??= encoderPath;

        // 5. 사용자 확인 (자동 선택된 것이 맞는지)
        if (decoderPath == encoderPath ||
            MessageBox.Show($"Use decoder: {Path.GetFileName(decoderPath)}?", "Confirm", MessageBoxButton.YesNo) == MessageBoxResult.No)
        {
            var dlg2 = new OpenFileDialog { Title = $"Select {CboModelType.Text} Decoder", Filter = "ONNX|*.onnx", InitialDirectory = folder };
            if (dlg2.ShowDialog(Window.GetWindow(this)) != true) return;
            decoderPath = dlg2.FileName;
        }

        SetBusy(true, "Loading models...");
        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;
            await Task.Run(() => _segmenter.LoadModels(encoderPath, decoderPath, useGpu));

            _isModelLoaded = true;
            TxtStatus.Text = $"{CboModelType.Text} Loaded ({_segmenter.DeviceMode})";
            BtnOpenImage.IsEnabled = true;
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error loading {CboModelType.Text}:\n{ex.Message}\n\nCheck if you selected the correct model files.");
            _isModelLoaded = false;
        }
        finally
        {
            SetBusy(false);
        }
    }

    private async void BtnOpenImage_Click(object sender, RoutedEventArgs e)
    {
        if (!_isModelLoaded) return;

        var dlg = new OpenFileDialog { Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp;*.webp" };
        if (dlg.ShowDialog(Window.GetWindow(this)) != true) return;

        // 1. 이미지 로드
        var bmp = new BitmapImage();
        bmp.BeginInit();
        bmp.UriSource = new Uri(dlg.FileName);
        bmp.CacheOption = BitmapCacheOption.OnLoad;
        bmp.EndInit();
        bmp.Freeze();

        // 2. DPI 정규화 (좌표 계산 및 마스크 정합성 확보를 위해 필수)
        _inputBitmap = NormalizeDpi96(bmp);

        ImgInput.Source = _inputBitmap;
        ImgMask.Source = null;
        PointOverlay.Children.Clear();
        _lastClickRatio = null;
        _isImageEncoded = false;
        TxtOverlay.Visibility = Visibility.Visible;

        // UI 초기화
        CboMaskCandidates.Items.Clear();
        CboMaskCandidates.IsEnabled = false;

        SetBusy(true, "Encoding...");
        try
        {
            byte[] bytes = BitmapToPngBytes(_inputBitmap);
            await Task.Run(() => _segmenter.EncodeImage(bytes));

            _isImageEncoded = true;
            TxtStatus.Text = "Ready. Click image.";
            TxtOverlay.Visibility = Visibility.Collapsed;

            BtnReset.IsEnabled = true;
        }
        catch (Exception ex)
        {
            MessageBox.Show(ex.ToString());
        }
        finally
        {
            SetBusy(false);
        }
    }

    private async void ImgInput_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (!_isImageEncoded || _inputBitmap == null) return;

        // 컨트롤 내 마우스 좌표
        Point viewPoint = e.GetPosition(ImgInput);

        // 실제 이미지가 렌더링된 영역 계산 (Letterbox 제외)
        Rect renderRect = GetImageRenderRect(ImgInput, _inputBitmap);
        if (!renderRect.Contains(viewPoint)) return;

        // 클릭 위치의 상대 비율(0.0 ~ 1.0) 계산
        double ratioX = (viewPoint.X - renderRect.X) / renderRect.Width;
        double ratioY = (viewPoint.Y - renderRect.Y) / renderRect.Height;
        _lastClickRatio = new Point(ratioX, ratioY);

        UpdateOverlayPoint();

        // 엔진에는 원본 이미지의 픽셀 좌표를 전달 (엔진 내부에서 Scale 계산)
        float targetX = (float)(ratioX * _inputBitmap.PixelWidth);
        float targetY = (float)(ratioY * _inputBitmap.PixelHeight);

        TxtStatus.Text = $"Segmenting {Math.Round(targetX)},{Math.Round(targetY)}...";
        try
        {
            // 인터페이스를 통한 예측 (MobileSAM 또는 SAM 2)
            var result = await Task.Run(() => _segmenter.Predict(targetX, targetY));

            if (result.Scores.Count > 0)
            {
                // 1. 1등 마스크 즉시 표시
                if (result.BestMaskBytes.Length > 0)
                {
                    var maskBmp = BytesToBitmap(result.BestMaskBytes);
                    ImgMask.Source = NormalizeDpi96(maskBmp);
                }

                // 2. 콤보박스 업데이트
                _isUpdatingCombo = true;
                CboMaskCandidates.SelectionChanged -= CboMaskCandidates_SelectionChanged;
                CboMaskCandidates.Items.Clear();
                CboMaskCandidates.IsEnabled = true;

                // 점수 내림차순 정렬된 인덱스 리스트 생성
                var sortedIndices = result.Scores
                                          .Select((s, i) => new { Score = s, Index = i })
                                          .OrderByDescending(x => x.Score)
                                          .ToList();

                foreach (var item in sortedIndices)
                {
                    var cbi = new ComboBoxItem
                    {
                        Content = $"Mask {item.Index + 1} ({item.Score:P1})",
                        Tag = item.Index
                    };
                    CboMaskCandidates.Items.Add(cbi);
                }

                // BestIndex 선택
                for (int i = 0; i < CboMaskCandidates.Items.Count; i++)
                {
                    if (CboMaskCandidates.Items[i] is ComboBoxItem item &&
                        (int)item.Tag == result.BestIndex)
                    {
                        CboMaskCandidates.SelectedIndex = i;
                        break;
                    }
                }

                // BestIndex가 없을 경우(거의 없음) 0번 선택
                if (CboMaskCandidates.SelectedIndex < 0)
                    CboMaskCandidates.SelectedIndex = 0;

                CboMaskCandidates.SelectionChanged += CboMaskCandidates_SelectionChanged;
                _isUpdatingCombo = false;

                TxtStatus.Text = "Done.";
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show(ex.ToString());
        }
    }

    private async void CboMaskCandidates_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        // 콤보박스 자동 세팅 중일 땐 이미지 생성 스킵
        if (_isUpdatingCombo) return;
        if (CboMaskCandidates.SelectedItem is not ComboBoxItem selectedItem) return;
        if (selectedItem.Tag is not int maskIndex) return;

        TxtStatus.Text = "Rendering Mask...";

        byte[] maskBytes = await Task.Run(() => _segmenter.GetMaskImage(maskIndex));

        if (maskBytes.Length > 0)
        {
            var maskBmp = BytesToBitmap(maskBytes);
            ImgMask.Source = NormalizeDpi96(maskBmp);
            TxtStatus.Text = $"Selected: {selectedItem.Content}";
        }
    }

    private void ImgInput_SizeChanged(object sender, SizeChangedEventArgs e) => UpdateOverlayPoint();

    /// <summary>
    /// 클릭한 지점에 빨간 점을 표시합니다. 
    /// </summary>
    private void UpdateOverlayPoint()
    {
        PointOverlay.Children.Clear();
        if (_lastClickRatio == null || _inputBitmap == null) return;

        Rect renderRect = GetImageRenderRect(ImgInput, _inputBitmap);

        // 비율 좌표를 현재 컨트롤 크기에 맞는 좌표로 변환
        double drawX = (_lastClickRatio.Value.X * renderRect.Width) + renderRect.X;
        double drawY = (_lastClickRatio.Value.Y * renderRect.Height) + renderRect.Y;

        Point p = ImgInput.TranslatePoint(new Point(drawX, drawY), PointOverlay);

        var ell = new Ellipse
        {
            Width = 10,
            Height = 10,
            Fill = Brushes.Red,
            Stroke = Brushes.White,
            StrokeThickness = 2
        };
        Canvas.SetLeft(ell, p.X - 5);
        Canvas.SetTop(ell, p.Y - 5);
        PointOverlay.Children.Add(ell);
    }

    /// <summary>
    /// Image 컨트롤 내부에서 실제 이미지가 그려지는 영역(Rect)을 계산합니다.
    /// (Uniform Stretch 모드에서 발생하는 검은 여백을 제외한 영역)
    /// </summary>
    private Rect GetImageRenderRect(Image imgControl, BitmapSource bmp)
    {
        double ctrlW = imgControl.ActualWidth;
        double ctrlH = imgControl.ActualHeight;
        double bmpW = bmp.Width;
        double bmpH = bmp.Height;

        if (ctrlW <= 0 || ctrlH <= 0 || bmpW <= 0 || bmpH <= 0) return Rect.Empty;

        double aspectControl = ctrlW / ctrlH;
        double aspectImage = bmpW / bmpH;

        double renderW, renderH;

        if (aspectControl > aspectImage) // 컨트롤이 더 넓음 (좌우 여백)
        {
            renderH = ctrlH;
            renderW = ctrlH * aspectImage;
        }
        else // 컨트롤이 더 높음 (상하 여백)
        {
            renderW = ctrlW;
            renderH = ctrlW / aspectImage;
        }

        double offsetX = (ctrlW - renderW) / 2.0;
        double offsetY = (ctrlH - renderH) / 2.0;

        return new Rect(offsetX, offsetY, renderW, renderH);
    }

    private void BtnReset_Click(object sender, RoutedEventArgs e)
    {
        ImgMask.Source = null;
        PointOverlay.Children.Clear();
        _lastClickRatio = null;

        CboMaskCandidates.Items.Clear();
        CboMaskCandidates.IsEnabled = false;

        TxtStatus.Text = "Mask cleared.";
    }

    private void SetBusy(bool busy, string? msg = null)
    {
        PbarLoading.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;
        BtnLoadModels.IsEnabled = !busy;
        BtnOpenImage.IsEnabled = !busy && _isModelLoaded;
        BtnReset.IsEnabled = !busy && _isModelLoaded;
        ImgInput.IsEnabled = !busy; // 처리 중 클릭 방지

        if (msg != null) TxtStatus.Text = msg;
    }

    private static byte[] BitmapToPngBytes(BitmapSource bmp)
    {
        var enc = new PngBitmapEncoder();
        enc.Frames.Add(BitmapFrame.Create(bmp));
        using var ms = new MemoryStream();
        enc.Save(ms);
        return ms.ToArray();
    }

    private static BitmapImage BytesToBitmap(byte[] bytes)
    {
        var img = new BitmapImage();
        using var ms = new MemoryStream(bytes);
        img.BeginInit();
        img.CacheOption = BitmapCacheOption.OnLoad;
        img.StreamSource = ms;
        img.EndInit();
        img.Freeze();
        return img;
    }

    /// <summary>
    /// 다양한 DPI를 가진 이미지를 96 DPI로 통일하여 좌표 계산 오차를 방지합니다.
    /// </summary>
    private static BitmapSource NormalizeDpi96(BitmapSource src)
    {
        if (src == null) throw new ArgumentNullException(nameof(src));
        const double dpi = 96.0;

        if (Math.Abs(src.DpiX - dpi) < 0.01 && Math.Abs(src.DpiY - dpi) < 0.01)
            return src;

        int w = src.PixelWidth;
        int h = src.PixelHeight;
        var pf = src.Format;

        if (pf == PixelFormats.Indexed1 || pf == PixelFormats.Indexed2 ||
            pf == PixelFormats.Indexed4 || pf == PixelFormats.Indexed8)
        {
            src = new FormatConvertedBitmap(src, PixelFormats.Bgra32, null, 0);
            pf = PixelFormats.Bgra32;
        }

        int bpp = (pf.BitsPerPixel + 7) / 8;
        int stride = w * bpp;
        byte[] buf = new byte[h * stride];
        src.CopyPixels(buf, stride, 0);

        var normalized = BitmapSource.Create(w, h, dpi, dpi, pf, null, buf, stride);
        normalized.Freeze();
        return normalized;
    }
}
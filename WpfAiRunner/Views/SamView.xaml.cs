using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using Microsoft.Win32;
using OnnxEngines.Sam;
using OnnxEngines.Utils;
using SamEngine;
using Path = System.IO.Path;

namespace WpfAiRunner.Views;

public partial class SamView : BaseAiView
{
    private ISamSegmenter _segmenter;
    private bool _isModelLoaded;
    private bool _isImageEncoded;
    private Point? _lastClickRatio;
    private bool _isUpdatingCombo;
    private string? _currentEncoderPath;
    private string? _currentDecoderPath;

    protected override Image ControlImgInput => ImgInput;
    protected override Image? ControlImgOutput => null; // SAM은 별도 오버레이 사용
    protected override ProgressBar? ControlPbarLoading => PbarLoading;
    protected override TextBlock? ControlTxtStatus => TxtStatus;

    public SamView()
    {
        InitializeComponent();
        _segmenter = new Sam2Segmenter();
    }

    public override void Dispose() => _segmenter?.Dispose();

    // 1. 화면 로드 시: 현재 선택된 모드에 맞춰 자동 로드 시도
    protected override async void OnLoaded(RoutedEventArgs e)
    {
        await AutoLoadModelForCurrentType();
    }

    // 2. 이미지가 로드되면 자동으로 인코딩 수행
    protected override async void OnImageLoaded()
    {
        // DPI 정규화
        if (_inputBitmap != null)
        {
            _inputBitmap = NormalizeDpi96(_inputBitmap);
            ImgInput.Source = _inputBitmap;
        }

        ImgMask.Source = null;
        PointOverlay.Children.Clear();
        _lastClickRatio = null;
        CboMaskCandidates.Items.Clear();
        CboMaskCandidates.IsEnabled = false;
        TxtOverlay.Visibility = Visibility.Visible;

        await EncodeCurrentInput();
    }

    // 모델 타입 변경 이벤트
    private async void CboModelType_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!IsLoaded) return;
        await AutoLoadModelForCurrentType();
    }

    private async Task AutoLoadModelForCurrentType()
    {
        int index = CboModelType.SelectedIndex;
        _segmenter.Dispose();
        _segmenter = (index == 0) ? (ISamSegmenter)new SamSegmenter() : new Sam2Segmenter();

        _isModelLoaded = false;
        _isImageEncoded = false;
        Log("Model changed. Please load weights.");

#if DEBUG
        string? encoder = null, decoder = null;
        if (index == 0)
        {
            encoder = OnnxHelper.FindModelInDebug("mobile_sam.encoder.onnx");
            decoder = OnnxHelper.FindModelInDebug("mobile_sam.decoder.onnx");
        }
        else
        {
            encoder = OnnxHelper.FindModelInDebug("sam2_hiera_small.encoder.onnx");
            decoder = OnnxHelper.FindModelInDebug("sam2_hiera_small.decoder.onnx");
        }
        if (encoder != null && decoder != null) await LoadModelsInternal(encoder, decoder);
#endif
    }

    // 모델 로드 버튼 클릭
    private async void BtnLoadModels_Click(object sender, RoutedEventArgs e)
    {
        int modelTypeIndex = CboModelType.SelectedIndex; // 0: MobileSAM, 1: SAM 2

        // 엔진 리셋
        _segmenter.Dispose();
        if (modelTypeIndex == 0) _segmenter = new SamSegmenter();
        else _segmenter = new Sam2Segmenter();

        var dlg = new OpenFileDialog { Title = $"Select {CboModelType.Text} Encoder", Filter = "ONNX|*.onnx" };
        if (dlg.ShowDialog(Window.GetWindow(this)) != true) return;

        string encoderPath = dlg.FileName;
        string folder = Path.GetDirectoryName(encoderPath)!;
        string encoderNameLower = Path.GetFileName(encoderPath).ToLower();

        string? decoderPath = null;
        var allDecoders = Directory.GetFiles(folder, "*decoder*.onnx");

        if (modelTypeIndex == 1) // SAM 2
        {
            string[] variants = { "tiny", "small", "base_plus", "large" };
            string? detectedVariant = variants.FirstOrDefault(v => encoderNameLower.Contains(v));

            if (!string.IsNullOrEmpty(detectedVariant))
            {
                decoderPath = allDecoders.FirstOrDefault(f =>
                    Path.GetFileName(f).ToLower().Contains(detectedVariant) &&
                    Path.GetFileName(f).ToLower().Contains("sam2"));
            }
            decoderPath ??= allDecoders.FirstOrDefault(f =>
            {
                string name = Path.GetFileName(f).ToLower();
                return name.Contains("sam2") || name.Contains("hiera");
            });
        }
        else // MobileSAM
        {
            decoderPath = allDecoders.FirstOrDefault(f => Path.GetFileName(f).ToLower().Contains("mobile"));
            decoderPath ??= allDecoders.FirstOrDefault(f =>
            {
                string name = Path.GetFileName(f).ToLower();
                return !name.Contains("sam2") && !name.Contains("hiera");
            });
        }

        decoderPath ??= encoderPath;

        if (decoderPath == encoderPath ||
            MessageBox.Show($"Use decoder: {Path.GetFileName(decoderPath)}?", "Confirm", MessageBoxButton.YesNo) == MessageBoxResult.No)
        {
            var dlg2 = new OpenFileDialog { Title = $"Select {CboModelType.Text} Decoder", Filter = "ONNX|*.onnx", InitialDirectory = folder };
            if (dlg2.ShowDialog(Window.GetWindow(this)) != true) return;
            decoderPath = dlg2.FileName;
        }

        await LoadModelsInternal(encoderPath, decoderPath);
    }

    // GPU 체크박스
    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e)
    {
        if (!_isModelLoaded || string.IsNullOrEmpty(_currentEncoderPath) || string.IsNullOrEmpty(_currentDecoderPath))
            return;
        await LoadModelsInternal(_currentEncoderPath, _currentDecoderPath);
    }

    private async Task LoadModelsInternal(string encoderPath, string decoderPath)
    {
        SetBusy(true, "Loading models...");
        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;
            await Task.Run(() => _segmenter.LoadModels(encoderPath, decoderPath, useGpu));

            _currentEncoderPath = encoderPath;
            _currentDecoderPath = decoderPath;
            _isModelLoaded = true;

            Log($"{((ComboBoxItem)CboModelType.SelectedItem).Content} Loaded ({_segmenter.DeviceMode})");

            if (useGpu && _segmenter.DeviceMode.Contains("CPU"))
            {
                ChkUseGpu.IsChecked = false;
                MessageBox.Show("GPU init failed. Fallback to CPU.", "Info");
            }

            if (_inputBitmap != null) await EncodeCurrentInput();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error loading models:\n{ex.Message}");
            _isModelLoaded = false;
            _currentEncoderPath = null;
            _currentDecoderPath = null;
        }
        finally
        {
            SetBusy(false);
        }
    }

    // 이미지 열기 버튼 (부모 메서드 호출)
    private void BtnOpenImage_Click(object sender, RoutedEventArgs e) => OpenImageDialog();

    // 리셋 버튼
    private void BtnReset_Click(object sender, RoutedEventArgs e)
    {
        ImgMask.Source = null;
        PointOverlay.Children.Clear();
        _lastClickRatio = null;
        CboMaskCandidates.Items.Clear();
        CboMaskCandidates.IsEnabled = false;
        Log("Mask cleared.");
    }

    private async Task EncodeCurrentInput()
    {
        if (_inputBitmap == null || !_isModelLoaded) return;
        SetBusy(true, "Encoding...");
        try
        {
            _isImageEncoded = false;
            byte[] bytes = BitmapToBytes(_inputBitmap);
            await Task.Run(() => _segmenter.EncodeImage(bytes));
            _isImageEncoded = true;
            Log("Ready. Click image.");
            TxtOverlay.Visibility = Visibility.Collapsed;
            BtnReset.IsEnabled = true;
        }
        catch (Exception ex) { MessageBox.Show($"Encoding failed: {ex.Message}"); }
        finally { SetBusy(false); }
    }

    // 이미지 클릭 이벤트
    private async void ImgInput_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (!_isImageEncoded || _inputBitmap == null) return;

        Point viewPoint = e.GetPosition(ImgInput);
        Rect renderRect = GetImageRenderRect(ImgInput, _inputBitmap);
        if (!renderRect.Contains(viewPoint)) return;

        double ratioX = (viewPoint.X - renderRect.X) / renderRect.Width;
        double ratioY = (viewPoint.Y - renderRect.Y) / renderRect.Height;
        _lastClickRatio = new Point(ratioX, ratioY);

        UpdateOverlayPoint();

        float targetX = (float)(ratioX * _inputBitmap.PixelWidth);
        float targetY = (float)(ratioY * _inputBitmap.PixelHeight);

        Log($"Segmenting {Math.Round(targetX)},{Math.Round(targetY)}...");
        try
        {
            var result = await Task.Run(() => _segmenter.Predict(targetX, targetY));

            if (result.Scores.Count > 0)
            {
                if (result.BestMaskBytes.Length > 0)
                {
                    var maskBmp = BytesToBitmap(result.BestMaskBytes);
                    ImgMask.Source = NormalizeDpi96(maskBmp);
                }

                _isUpdatingCombo = true;
                CboMaskCandidates.SelectionChanged -= CboMaskCandidates_SelectionChanged;
                CboMaskCandidates.Items.Clear();
                CboMaskCandidates.IsEnabled = true;

                var sortedIndices = result.Scores.Select((s, i) => new { Score = s, Index = i })
                                          .OrderByDescending(x => x.Score).ToList();

                foreach (var item in sortedIndices)
                {
                    CboMaskCandidates.Items.Add(new ComboBoxItem
                    {
                        Content = $"Mask {item.Index + 1} ({item.Score:P1})",
                        Tag = item.Index
                    });
                }

                for (int i = 0; i < CboMaskCandidates.Items.Count; i++)
                {
                    if (CboMaskCandidates.Items[i] is ComboBoxItem item && (int)item.Tag == result.BestIndex)
                    {
                        CboMaskCandidates.SelectedIndex = i;
                        break;
                    }
                }
                if (CboMaskCandidates.SelectedIndex < 0) CboMaskCandidates.SelectedIndex = 0;

                CboMaskCandidates.SelectionChanged += CboMaskCandidates_SelectionChanged;
                _isUpdatingCombo = false;
                Log("Done.");
            }
        }
        catch (Exception ex) { MessageBox.Show(ex.ToString()); }
    }

    // 마스크 후보 선택 변경
    private async void CboMaskCandidates_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_isUpdatingCombo) return;
        if (CboMaskCandidates.SelectedItem is not ComboBoxItem selectedItem) return;
        if (selectedItem.Tag is not int maskIndex) return;

        Log("Rendering Mask...");
        byte[] maskBytes = await Task.Run(() => _segmenter.GetMaskImage(maskIndex));
        if (maskBytes.Length > 0)
        {
            var maskBmp = BytesToBitmap(maskBytes);
            ImgMask.Source = NormalizeDpi96(maskBmp);
            Log($"Selected: {selectedItem.Content}");
        }
    }

    // 이미지 크기 변경 시 점 위치 업데이트
    private void ImgInput_SizeChanged(object sender, SizeChangedEventArgs e) => UpdateOverlayPoint();

    private void UpdateOverlayPoint()
    {
        PointOverlay.Children.Clear();
        if (_lastClickRatio == null || _inputBitmap == null) return;

        Rect renderRect = GetImageRenderRect(ImgInput, _inputBitmap);
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
        if (aspectControl > aspectImage) { renderH = ctrlH; renderW = ctrlH * aspectImage; }
        else { renderW = ctrlW; renderH = ctrlW / aspectImage; }

        double offsetX = (ctrlW - renderW) / 2.0;
        double offsetY = (ctrlH - renderH) / 2.0;

        return new Rect(offsetX, offsetY, renderW, renderH);
    }

    private void SetBusy(bool busy, string? msg = null)
    {
        SetBusyState(busy);
        if (msg != null) Log(msg);
        BtnLoadModels.IsEnabled = !busy;
        ChkUseGpu.IsEnabled = !busy;
        BtnOpenImage.IsEnabled = !busy && _isModelLoaded;
        BtnReset.IsEnabled = !busy && _isModelLoaded;
        ImgInput.IsEnabled = !busy;
    }

    private static BitmapSource NormalizeDpi96(BitmapSource src)
    {
        if (src == null) throw new ArgumentNullException(nameof(src));
        const double dpi = 96.0;
        if (Math.Abs(src.DpiX - dpi) < 0.01 && Math.Abs(src.DpiY - dpi) < 0.01) return src;

        int w = src.PixelWidth;
        int h = src.PixelHeight;
        var pf = src.Format;

        if (pf == PixelFormats.Indexed1 || pf == PixelFormats.Indexed2 || pf == PixelFormats.Indexed4 || pf == PixelFormats.Indexed8)
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
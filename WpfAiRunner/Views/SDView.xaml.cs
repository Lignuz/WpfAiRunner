using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using Microsoft.Win32;
using OnnxEngines.Utils;
using SdGgufEngine;
using Path = System.IO.Path;

namespace WpfAiRunner.Views;

public partial class SdView : BaseAiView
{
    private SdEngine _engine = new SdEngine();
    private string? _modelPath;
    private BitmapSource? _outputBitmap;

    // 마스크 관련
    private byte[]? _maskGray8;
    private int _maskW, _maskH;
    private bool _dragging, _painting;
    private Point _dragStart;
    private Rectangle? _rectVisual;
    private int _brushRadiusPx = 12;
    private bool _overlayHover;
    private bool _busy;

    protected override Image ControlImgInput => ImgInput;
    protected override Image? ControlImgOutput => ImgOutput;
    protected override ProgressBar? ControlPbarLoading => PbarLoading;
    protected override TextBlock? ControlTxtStatus => TxtStatus;

    public SdView()
    {
        InitializeComponent();
        _engine.LogAction = (msg) => Dispatcher.Invoke(() => Log(msg));
        _engine.ProgressAction = (s, t, p) => Dispatcher.Invoke(() =>
        {
            Log($"Step {s}/{t} ({p:P0})");
            if (ControlPbarLoading != null)
            {
                ControlPbarLoading.IsIndeterminate = false;
                ControlPbarLoading.Value = p * 100;
            }
        });
    }

    public override void Dispose() => _engine.Dispose();

    protected override async void OnLoaded(RoutedEventArgs e)
    {
        _engine.Initialize();
        if (RbRect != null) RbRect.IsChecked = true;

#if DEBUG
        string? debugPath = OnnxHelper.FindModelInDebug("stable-diffusion-v1-5-Q8_0.gguf");
        if (debugPath != null) await ReloadModelInternal(debugPath);
#endif
    }

    protected override void OnImageLoaded()
    {
        EnsureMaskBuffer();
        ClearMaskInternal();
        Log("Image Loaded / Mask Reset");
    }

    // 모델 선택 버튼
    private async void BtnPickModel_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "GGUF (*.gguf)|*.gguf" };
        if (dlg.ShowDialog(Window.GetWindow(this)) != true) return;
        await ReloadModelInternal(dlg.FileName);
    }

    // 모드 변경
    private void CmbMode_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!IsLoaded) return;
        if (CmbMode.SelectedItem is ComboBoxItem item)
        {
            string mode = item.Tag?.ToString() ?? "TextToImage";
            bool needsInput = (mode == "ImageToImage" || mode == "Inpainting");

            PanelImageButtons.Visibility = needsInput ? Visibility.Visible : Visibility.Collapsed;
            PanelMaskTools.Visibility = (mode == "Inpainting") ? Visibility.Visible : Visibility.Collapsed;
            PanelStrength.Visibility = needsInput ? Visibility.Visible : Visibility.Collapsed;
            BtnClearMask.Visibility = (mode == "Inpainting") ? Visibility.Visible : Visibility.Collapsed;

            CanvasMask.IsHitTestVisible = (mode == "Inpainting");
            CanvasMask.Background = (mode == "Inpainting") ? Brushes.Transparent : null;

            switch (mode)
            {
                case "TextToImage": TxtPrompt.Text = "a cute cat sitting on a wooden table, high quality, detailed"; break;
                case "ImageToImage": TxtPrompt.Text = "oil painting style, artistic, vibrant colors"; break;
                case "Inpainting": TxtPrompt.Text = "a beautiful flower"; break;
            }
        }
    }

    // 이미지 열기 버튼 (부모 메서드 호출)
    private void BtnLoadImage_Click(object sender, RoutedEventArgs e) => OpenImageDialog();

    // 결과물 사용 버튼
    private void BtnUseOutput_Click(object sender, RoutedEventArgs e)
    {
        if (_outputBitmap == null) return;
        _inputBitmap = _outputBitmap;
        ImgInput.Source = _inputBitmap;
        BaseAiView.SharedImage = _inputBitmap;
        EnsureMaskBuffer();
        ClearMaskInternal();
        Log("Output image set as input");
    }

    // 마스크 지우기
    private void BtnClearMask_Click(object sender, RoutedEventArgs e)
    {
        ClearMaskInternal();
        Log("Mask cleared");
    }

    // 마스크 모드 변경 (Rect vs Brush)
    private void MaskMode_Checked(object sender, RoutedEventArgs e)
    {
        bool isBrush = RbBrush?.IsChecked == true;
        PanelBrushSize.Visibility = isBrush ? Visibility.Visible : Visibility.Collapsed;
    }

    // 슬라이더 이벤트
    private void SliderStrength_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (TxtStrengthValue != null) TxtStrengthValue.Text = e.NewValue.ToString("F2");
    }

    private void SliderBrushSize_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        int size = (int)e.NewValue;
        _brushRadiusPx = Math.Max(1, size / 2);
        if (TxtBrushSizeDisplay != null) TxtBrushSizeDisplay.Text = size.ToString();
    }

    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(_modelPath)) return;

        var selectedMode = (CmbMode.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "TextToImage";
        GenerationMode mode = selectedMode switch
        {
            "ImageToImage" => GenerationMode.ImageToImage,
            "Inpainting" => GenerationMode.Inpainting,
            _ => GenerationMode.TextToImage
        };

        if (mode != GenerationMode.TextToImage && _inputBitmap == null)
        {
            MessageBox.Show("Please load an input image first.");
            return;
        }

        SetBusy(true, "Generating...");
        TxtPlaceholder.Visibility = Visibility.Collapsed;
        if (ControlPbarLoading != null) ControlPbarLoading.Value = 0;

        var param = new ModelParameter
        {
            Mode = mode,
            Prompt = TxtPrompt.Text,
            NegativePrompt = TxtNegative.Text,
            Steps = int.TryParse(TxtSteps.Text, out int s) ? s : 20,
            Cfg = float.TryParse(TxtCfg.Text, out float c) ? c : 7.5f,
            Seed = long.TryParse(TxtSeed.Text, out long sd) ? sd : -1,
            Width = 512,
            Height = 512,

            InputImage = _inputBitmap != null ? BitmapToBytes(_inputBitmap) : null,
            MaskImage = (mode == GenerationMode.Inpainting && _maskGray8 != null) ? Gray8ToPngBytes(_maskGray8, _maskW, _maskH) : null,
            Strength = (float)SliderStrength.Value
        };

        try
        {
            byte[]? result = await Task.Run(() => _engine.Generate(param));
            if (result != null)
            {
                _outputBitmap = BytesToBitmap(result);
                ImgOutput.Source = _outputBitmap;
                BtnUseOutput.IsEnabled = true;
                Log($"Done! ({mode})");
            }
        }
        catch (Exception ex) { MessageBox.Show($"Error: {ex.Message}"); }
        finally { SetBusy(false); GC.Collect(); }
    }

    // 캔버스 마우스 이벤트들
    private void CanvasMask_MouseEnter(object sender, MouseEventArgs e)
    {
        _overlayHover = true;
        UpdateBrushCursor(e.GetPosition(CanvasMask));
    }

    private void CanvasMask_MouseLeave(object sender, MouseEventArgs e)
    {
        _overlayHover = false;
        BrushCursor.Visibility = Visibility.Collapsed;
    }

    private void CanvasMask_MouseDown(object sender, MouseButtonEventArgs e)
    {
        if (_busy || _inputBitmap == null) return;
        CanvasMask.CaptureMouse();

        if (RbRect.IsChecked == true)
        {
            _dragging = true;
            _dragStart = e.GetPosition(CanvasMask);
            if (_rectVisual == null)
            {
                _rectVisual = new Rectangle { Stroke = Brushes.Cyan, StrokeThickness = 2, Fill = Brushes.Transparent, StrokeDashArray = new DoubleCollection { 3, 2 } };
                CanvasMask.Children.Add(_rectVisual);
            }
            Canvas.SetLeft(_rectVisual, _dragStart.X);
            Canvas.SetTop(_rectVisual, _dragStart.Y);
            _rectVisual.Width = 0; _rectVisual.Height = 0; _rectVisual.Visibility = Visibility.Visible;
        }
        else
        {
            _painting = true;
            PaintBrushAt(e.GetPosition(CanvasMask));
        }
    }

    private void CanvasMask_MouseMove(object sender, MouseEventArgs e)
    {
        if (_inputBitmap == null) return;
        var pt = e.GetPosition(CanvasMask);

        if (RbRect.IsChecked == true && _dragging && _rectVisual != null)
        {
            UpdateRectVisual(_dragStart, pt);
        }
        else if (RbBrush.IsChecked == true)
        {
            UpdateBrushCursor(pt);
            if (_painting && e.LeftButton == MouseButtonState.Pressed && !_busy)
                PaintBrushAt(pt);
        }
    }

    private void CanvasMask_MouseUp(object sender, MouseButtonEventArgs e)
    {
        if (_busy || _inputBitmap == null) return;
        CanvasMask.ReleaseMouseCapture();

        if (RbRect.IsChecked == true && _dragging)
        {
            _dragging = false;
            ApplyRectMask(_dragStart, e.GetPosition(CanvasMask));
            if (_rectVisual != null) _rectVisual.Visibility = Visibility.Collapsed;
        }
        else
        {
            _painting = false;
        }
    }

    // --- 내부 로직 ---

    private async Task ReloadModelInternal(string path)
    {
        _modelPath = path; SetBusy(true, "Loading Model...");
        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;
            bool success = await Task.Run(() => _engine.LoadModel(path, useGpu));
            if (success)
            {
                TxtModel.Text = Path.GetFileName(path);
                BtnRun.IsEnabled = true;
                Log($"Loaded on {(useGpu ? "GPU" : "CPU")}");
            }
            else { MessageBox.Show("Failed to load model."); Log("Load failed."); }
        }
        catch (Exception ex) { MessageBox.Show($"Error: {ex.Message}"); }
        finally { SetBusy(false); }
    }

    private void SetBusy(bool busy, string? status = null)
    {
        _busy = busy;
        SetBusyState(busy);
        if (status != null) Log(status);
        BtnRun.IsEnabled = !busy && _modelPath != null;
        BtnPickModel.IsEnabled = !busy; BtnLoadImage.IsEnabled = !busy;
    }

    private void EnsureMaskBuffer()
    {
        if (_inputBitmap == null) return;
        _maskW = _inputBitmap.PixelWidth; _maskH = _inputBitmap.PixelHeight;
        _maskGray8 = new byte[_maskW * _maskH];
    }

    private void ClearMaskInternal()
    {
        if (_maskGray8 == null) return;
        Array.Clear(_maskGray8, 0, _maskGray8.Length);
        if (_rectVisual != null) _rectVisual.Visibility = Visibility.Collapsed;
        UpdateMaskPreview();
    }

    private void UpdateMaskPreview()
    {
        if (_maskGray8 == null) { ImgMaskPreview.Source = null; return; }
        int stride = _maskW * 4;
        byte[] bgra = new byte[_maskW * _maskH * 4];
        for (int i = 0; i < _maskGray8.Length; i++) { int j = i * 4; bgra[j + 2] = 255; bgra[j + 3] = _maskGray8[i]; }
        var bmp = BitmapSource.Create(_maskW, _maskH, 96, 96, PixelFormats.Bgra32, null, bgra, stride);
        bmp.Freeze();
        ImgMaskPreview.Source = bmp;
    }

    private void UpdateRectVisual(Point a, Point b)
    {
        if (_rectVisual == null) return;
        double x1 = Math.Min(a.X, b.X), y1 = Math.Min(a.Y, b.Y);
        double w = Math.Abs(a.X - b.X), h = Math.Abs(a.Y - b.Y);
        Canvas.SetLeft(_rectVisual, x1); Canvas.SetTop(_rectVisual, y1);
        _rectVisual.Width = w; _rectVisual.Height = h;
    }

    private void ApplyRectMask(Point a, Point b)
    {
        if (_inputBitmap == null || _maskGray8 == null) return;
        var r = CanvasRectToImageRect(a, b);
        if (r.w <= 0 || r.h <= 0) return;
        for (int y = r.y; y < r.y + r.h; y++)
        {
            int row = y * _maskW;
            for (int x = r.x; x < r.x + r.w; x++) _maskGray8[row + x] = 255;
        }
        UpdateMaskPreview();
    }

    private void PaintBrushAt(Point ptCanvas)
    {
        if (_inputBitmap == null || _maskGray8 == null) return;
        var p = CanvasPointToImagePoint(ptCanvas);
        if (p.x < 0) return;
        int cx = p.x, cy = p.y, r = _brushRadiusPx, r2 = r * r;
        int y0 = Math.Max(0, cy - r), y1 = Math.Min(_maskH - 1, cy + r);
        int x0 = Math.Max(0, cx - r), x1 = Math.Min(_maskW - 1, cx + r);
        for (int y = y0; y <= y1; y++)
        {
            int row = y * _maskW, dy = y - cy;
            for (int x = x0; x <= x1; x++) { int dx = x - cx; if (dx * dx + dy * dy <= r2) _maskGray8[row + x] = 255; }
        }
        UpdateMaskPreview();
    }

    private void UpdateBrushCursor(Point? pt)
    {
        if (!_overlayHover || _busy || _inputBitmap == null || RbBrush?.IsChecked != true) { BrushCursor.Visibility = Visibility.Collapsed; return; }
        Point p = pt ?? Mouse.GetPosition(CanvasMask);
        double cw = CanvasMask.ActualWidth, ch = CanvasMask.ActualHeight;
        if (cw <= 2) return;
        int iw = _inputBitmap.PixelWidth, ih = _inputBitmap.PixelHeight;
        double imgAspect = (double)iw / ih, canvasAspect = cw / ch;
        double dispW = (canvasAspect > imgAspect) ? ch * imgAspect : cw;
        double dispH = (canvasAspect > imgAspect) ? ch : cw / imgAspect;
        double offX = (cw - dispW) / 2, offY = (ch - dispH) / 2;
        if (p.X < offX || p.X > offX + dispW || p.Y < offY || p.Y > offY + dispH) { BrushCursor.Visibility = Visibility.Collapsed; return; }
        double scale = dispW / iw;
        double rDisp = Math.Max(2.0, _brushRadiusPx * scale);
        BrushCursor.Width = rDisp * 2; BrushCursor.Height = rDisp * 2;
        Canvas.SetLeft(BrushCursor, p.X - rDisp); Canvas.SetTop(BrushCursor, p.Y - rDisp);
        BrushCursor.Visibility = Visibility.Visible;
    }

    private (int x, int y) CanvasPointToImagePoint(Point pt)
    {
        if (_inputBitmap == null) return (-1, -1);
        double cw = CanvasMask.ActualWidth, ch = CanvasMask.ActualHeight;
        int iw = _inputBitmap.PixelWidth, ih = _inputBitmap.PixelHeight;
        double dispW = (cw / ch > (double)iw / ih) ? ch * iw / ih : cw;
        double dispH = (cw / ch > (double)iw / ih) ? ch : cw / ((double)iw / ih);
        double offX = (cw - dispW) / 2, offY = (ch - dispH) / 2;
        int px = (int)Math.Round((pt.X - offX) / dispW * (iw - 1));
        int py = (int)Math.Round((pt.Y - offY) / dispH * (ih - 1));
        return (px < 0 || px >= iw || py < 0 || py >= ih) ? (-1, -1) : (px, py);
    }

    private (int x, int y, int w, int h) CanvasRectToImageRect(Point a, Point b)
    {
        var p1 = CanvasPointToImagePoint(a); var p2 = CanvasPointToImagePoint(b);
        if (p1.x == -1 || p2.x == -1) return (0, 0, 0, 0);
        int x1 = Math.Min(p1.x, p2.x), y1 = Math.Min(p1.y, p2.y);
        int x2 = Math.Max(p1.x, p2.x), y2 = Math.Max(p1.y, p2.y);
        return (x1, y1, x2 - x1, y2 - y1);
    }

    private static byte[] Gray8ToPngBytes(byte[] g, int w, int h)
    {
        var b = BitmapSource.Create(w, h, 96, 96, PixelFormats.Gray8, null, g, w);
        var e = new PngBitmapEncoder(); e.Frames.Add(BitmapFrame.Create(b));
        using var m = new MemoryStream(); e.Save(m); return m.ToArray();
    }
}
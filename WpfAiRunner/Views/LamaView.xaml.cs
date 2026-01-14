using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using Microsoft.Win32;
using OnnxEngines.Lama;
using OnnxEngines.Utils;

namespace WpfAiRunner.Views;

public partial class LamaView : BaseAiView
{
    private LamaInpainter? _inpainter;
    private string? _modelPath;
    private BitmapSource? _outputBitmap; // 결과물 보관용 변수 

    private byte[]? _maskGray8;
    private int _maskW;
    private int _maskH;

    private enum MaskMode { Rect, Brush }
    private MaskMode _maskMode = MaskMode.Rect;
    private bool _dragging;
    private Point _dragStart;
    private Rectangle? _rectVisual;
    private bool _painting;
    private int _brushRadiusPx = 12;
    private bool _uiReady;
    private bool _busy;
    private bool _overlayHover;

    protected override Image ControlImgInput => ImgInput;
    protected override Image? ControlImgOutput => ImgOutput;
    protected override ProgressBar? ControlPbarLoading => PbarLoading;
    protected override TextBlock? ControlTxtStatus => TxtStatus;

    public LamaView()
    {
        InitializeComponent();
    }

    public override void Dispose() => _inpainter?.Dispose();

    protected override async void OnLoaded(RoutedEventArgs e)
    {
        rbRect.Checked += MaskMode_Checked;
        rbBrush.Checked += MaskMode_Checked;
        SlBrushSize.ValueChanged += SlBrushSize_ValueChanged;
        TxtBrushSize.TextChanged += TxtBrushSize_TextChanged;

        rbRect.IsChecked = true;
        _uiReady = true;
        UpdateBrushUi();
        UpdateButtons();
        Log("Ready.");

#if DEBUG
        if (_inpainter == null && string.IsNullOrEmpty(_modelPath))
        {
            string? debugPath = OnnxHelper.FindModelInDebug("lama_fp32.onnx");
            if (debugPath != null) await ReloadModelAsync(debugPath);
        }
#endif
    }

    protected override void OnImageLoaded()
    {
        EnsureMaskBufferForInput();
        ClearMaskInternal();
        UpdateButtons();
    }

    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e)
    {
        if (!string.IsNullOrEmpty(_modelPath)) await ReloadModelAsync(_modelPath);
    }

    private async void BtnPickModel_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "ONNX model (*.onnx)|*.onnx" };
        if (dlg.ShowDialog(Window.GetWindow(this)) != true) return;
        await ReloadModelAsync(dlg.FileName);
    }

    private async Task ReloadModelAsync(string path)
    {
        SetBusy(true);
        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;
            _inpainter?.Dispose();
            _inpainter = await Task.Run(() => new LamaInpainter(path, useGpu));
            _modelPath = path;
            TxtModel.Text = System.IO.Path.GetFileName(_modelPath);
            Log($"Model loaded on {_inpainter.DeviceMode}.");

            if (useGpu && _inpainter.DeviceMode.Contains("CPU")) ChkUseGpu.IsChecked = false;
        }
        catch (Exception ex)
        {
            MessageBox.Show(ex.ToString());
            _inpainter = null;
            TxtModel.Text = "(no model)";
            Log("Model load failed.");
        }
        finally
        {
            SetBusy(false);
            UpdateButtons();
        }
    }

    private void BtnOpenImage_Click(object sender, RoutedEventArgs e) => OpenImageDialog();

    // 결과물을 입력으로 사용
    private void BtnUseOutput_Click(object sender, RoutedEventArgs e)
    {
        if (_outputBitmap == null) return;

        // 1. 입력을 결과물로 교체
        _inputBitmap = _outputBitmap;
        ImgInput.Source = _inputBitmap;

        // 2. 다른 뷰와 공유하기 위해 업데이트
        BaseAiView.SharedImage = _inputBitmap;

        // 3. 마스크 초기화 (새 이미지이므로)
        EnsureMaskBufferForInput();
        ClearMaskInternal();

        Log("Output image set as input.");
    }

    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        if (_inpainter == null || _inputBitmap == null || _maskGray8 == null) return;
        if (!HasAnyMask()) { Log("No mask."); return; }

        SetBusy(true, "Running inference...");
        try
        {
            byte[] inputPng = BitmapToBytes(_inputBitmap);
            byte[] maskPng = Gray8MaskToPngBytes(_maskGray8, _maskW, _maskH);
            byte[] outPng = await Task.Run(() => _inpainter!.ProcessImage(inputPng, maskPng));

            // 결과물 표시 및 보관
            _outputBitmap = BytesToBitmap(outPng);
            ImgOutput.Source = _outputBitmap;

            Log("Done.");
        }
        catch (Exception ex)
        {
            MessageBox.Show(ex.ToString());
            Log("Inference failed.");
        }
        finally
        {
            SetBusy(false);
            GC.Collect();
        }
        UpdateButtons();
    }

    private void BtnClearMask_Click(object sender, RoutedEventArgs e)
    {
        ClearMaskInternal();
        UpdateButtons();
        Log("Mask cleared.");
    }

    // --- Mask UI Logic ---
    void MaskMode_Checked(object sender, RoutedEventArgs e)
    {
        if (!_uiReady) return;
        _maskMode = (rbBrush.IsChecked == true) ? MaskMode.Brush : MaskMode.Rect;
        UpdateBrushUi();
    }
    void UpdateBrushUi()
    {
        if (!_uiReady) return;
        bool brush = (rbBrush.IsChecked == true);
        TxtBrushSizeLabel.Opacity = brush ? 1.0 : 0.4;
        SlBrushSize.IsEnabled = brush && !_busy;
        TxtBrushSize.IsEnabled = brush && !_busy;
        int size = Clamp((int)Math.Round(SlBrushSize.Value), 4, 256);
        _brushRadiusPx = Math.Max(1, size / 2);
        UpdateBrushCursorVisual(null);
    }
    private void SlBrushSize_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e) => UpdateBrushUi();
    private void TxtBrushSize_TextChanged(object sender, TextChangedEventArgs e)
    {
        if (!_uiReady) return;
        if (int.TryParse(TxtBrushSize.Text, out int size)) SlBrushSize.Value = size;
    }

    // --- Mouse Interaction ---
    private void Overlay_MouseEnter(object sender, MouseEventArgs e) { _overlayHover = true; UpdateBrushCursorVisual(e.GetPosition(Overlay)); }
    private void Overlay_MouseLeave(object sender, MouseEventArgs e) { _overlayHover = false; UpdateBrushCursorVisual(null); }
    private void Overlay_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (_busy || _inputBitmap == null) return;
        Overlay.CaptureMouse();
        if (_maskMode == MaskMode.Rect) { _dragging = true; _dragStart = e.GetPosition(Overlay); StartRectVisual(); }
        else { _painting = true; PaintBrushAt(e.GetPosition(Overlay)); }
    }
    private void Overlay_MouseMove(object sender, MouseEventArgs e)
    {
        if (_inputBitmap == null) return;
        var pt = e.GetPosition(Overlay);
        if (_maskMode == MaskMode.Rect && _dragging && _rectVisual != null) UpdateRectVisual(_dragStart, pt);
        else if (_maskMode == MaskMode.Brush) { UpdateBrushCursorVisual(pt); if (_painting && e.LeftButton == MouseButtonState.Pressed && !_busy) PaintBrushAt(pt); }
    }
    private void Overlay_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        if (_busy || _inputBitmap == null) return;
        Overlay.ReleaseMouseCapture();
        if (_maskMode == MaskMode.Rect && _dragging) { _dragging = false; ApplyRectMask(_dragStart, e.GetPosition(Overlay)); if (_rectVisual != null) _rectVisual.Visibility = Visibility.Collapsed; }
        else _painting = false;
        UpdateButtons();
    }

    // --- Helper Methods ---
    private void StartRectVisual()
    {
        if (_rectVisual == null)
        {
            _rectVisual = new Rectangle { Stroke = Brushes.Cyan, StrokeThickness = 2, Fill = Brushes.Transparent, StrokeDashArray = new DoubleCollection { 3, 2 } };
            Overlay.Children.Add(_rectVisual);
        }
        Canvas.SetLeft(_rectVisual, _dragStart.X); Canvas.SetTop(_rectVisual, _dragStart.Y); _rectVisual.Width = 0; _rectVisual.Height = 0; _rectVisual.Visibility = Visibility.Visible;
    }

    private void EnsureMaskBufferForInput()
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
        UpdateMaskPreview(); UpdateBrushCursorVisual(null);
    }
    private bool HasAnyMask() => _maskGray8 != null && _maskGray8.Any(b => b != 0);

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

    private (int x, int y) CanvasPointToImagePoint(Point pt)
    {
        if (_inputBitmap == null) return (-1, -1);
        double cw = Overlay.ActualWidth, ch = Overlay.ActualHeight;
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

    private void UpdateRectVisual(Point a, Point b)
    {
        if (_rectVisual == null) return;
        double x1 = Math.Min(a.X, b.X), y1 = Math.Min(a.Y, b.Y);
        double w = Math.Abs(a.X - b.X), h = Math.Abs(a.Y - b.Y);
        Canvas.SetLeft(_rectVisual, x1); Canvas.SetTop(_rectVisual, y1);
        _rectVisual.Width = w; _rectVisual.Height = h;
        _rectVisual.Visibility = Visibility.Visible;
    }

    private void UpdateBrushCursorVisual(Point? pt)
    {
        if (!_overlayHover || _busy || _inputBitmap == null || rbBrush.IsChecked != true) { BrushCursor.Visibility = Visibility.Collapsed; return; }
        Point p = pt ?? Mouse.GetPosition(Overlay);
        double cw = Overlay.ActualWidth, ch = Overlay.ActualHeight;
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

    private void ApplyRectMask(Point a, Point b)
    {
        if (_inputBitmap == null || _maskGray8 == null) return;
        var r = CanvasRectToImageRect(a, b);
        if (r.w <= 0 || r.h <= 0) return;
        for (int y = r.y; y < r.y + r.h; y++) { int row = y * _maskW; for (int x = r.x; x < r.x + r.w; x++) _maskGray8[row + x] = 255; }
        UpdateMaskPreview();
    }

    private static int Clamp(int v, int min, int max) => (v < min) ? min : (v > max ? max : v);
    private static byte[] Gray8MaskToPngBytes(byte[] g, int w, int h)
    {
        var b = BitmapSource.Create(w, h, 96, 96, PixelFormats.Gray8, null, g, w);
        var e = new PngBitmapEncoder(); e.Frames.Add(BitmapFrame.Create(b));
        using var m = new MemoryStream(); e.Save(m); return m.ToArray();
    }

    private void SetBusy(bool busy, string? msg = null)
    {
        _busy = busy;
        SetBusyState(busy);
        if (msg != null) Log(msg);
        // ChkUseGpu.IsEnabled = !busy; BtnPickModel.IsEnabled = !busy; BtnOpenImage.IsEnabled = !busy; // not support GPU
        BtnUseOutput.IsEnabled = !busy && _outputBitmap != null; // 버튼 상태 관리
        BtnRun.IsEnabled = !busy && _inpainter != null && _inputBitmap != null && HasAnyMask();
        BtnClearMask.IsEnabled = !busy && _inputBitmap != null && HasAnyMask();
        rbRect.IsEnabled = !busy; rbBrush.IsEnabled = !busy; SlBrushSize.IsEnabled = !busy && rbBrush.IsChecked == true;
    }
    private void UpdateButtons() => SetBusy(_busy, null);
}
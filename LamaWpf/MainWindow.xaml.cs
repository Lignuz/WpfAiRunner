using System.IO;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using LamaEngine;
using System.Windows.Controls;

namespace LamaWpf;

public partial class MainWindow : Window
{
    private LamaInpainter? _inpainter;
    private BitmapSource? _inputBitmap;

    // mask buffer (Gray8, 0=none, 255=masked)
    private byte[]? _maskGray8;
    private int _maskW;
    private int _maskH;

    private enum MaskMode { Rect, Brush }
    private MaskMode _maskMode = MaskMode.Rect;

    // rect drag
    private bool _dragging;
    private Point _dragStart;
    private System.Windows.Shapes.Rectangle? _rectVisual;

    // brush paint
    private bool _painting;
    private int _brushRadiusPx = 12;

    // ui/state
    private bool _uiReady;
    private bool _busy;
    private bool _overlayHover;

    public MainWindow()
    {
        InitializeComponent();
    }

    private void Window_Loaded(object sender, RoutedEventArgs e)
    {
        // Wire events here to avoid Checked firing before fields are ready.
        rbRect.Checked += MaskMode_Checked;
        rbBrush.Checked += MaskMode_Checked;
        SlBrushSize.ValueChanged += SlBrushSize_ValueChanged;
        TxtBrushSize.TextChanged += TxtBrushSize_TextChanged;

        rbRect.IsChecked = true;
        rbBrush.IsChecked = false;

        _uiReady = true;
        UpdateBrushUi();
        UpdateButtons();
        SetStatus("Ready.");
    }

    private async void BtnPickModel_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Filter = "ONNX model (*.onnx)|*.onnx|All files (*.*)|*.*"
        };

        if (dlg.ShowDialog(this) != true) return;

        SetBusy(true, "Loading model...");

        try
        {
            // Heavy init can stutter UI -> run on worker thread.
            _inpainter?.Dispose();
            var path = dlg.FileName;

            _inpainter = await Task.Run(() => new LamaInpainter(path));

            TxtModel.Text = System.IO.Path.GetFileName(path);
            SetStatus("Model loaded.");
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.ToString(), "Model load failed", MessageBoxButton.OK, MessageBoxImage.Error);
            _inpainter = null;
            TxtModel.Text = "(no model)";
            SetStatus("Model load failed.");
        }
        finally
        {
            SetBusy(false, null);
        }

        UpdateButtons();
    }

    private void BtnOpenImage_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Filter = "Image (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp|All files (*.*)|*.*"
        };

        if (dlg.ShowDialog(this) != true) return;

        try
        {
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.CacheOption = BitmapCacheOption.OnLoad;
            bmp.UriSource = new Uri(dlg.FileName);
            bmp.EndInit();
            bmp.Freeze();

            _inputBitmap = bmp;
            ImgInput.Source = _inputBitmap;

            EnsureMaskBufferForInput();
            ClearMaskInternal(); // new input -> clear mask

            SetStatus("Image loaded.");
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.ToString(), "Open image failed", MessageBoxButton.OK, MessageBoxImage.Error);
        }

        UpdateButtons();
    }

    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        if (_inpainter == null || _inputBitmap == null || _maskGray8 == null) return;
        if (!HasAnyMask()) { SetStatus("No mask."); return; }

        SetBusy(true, "Running inference...");

        try
        {
            byte[] inputPng = BitmapToPngBytes(_inputBitmap);
            byte[] maskPng = Gray8MaskToPngBytes(_maskGray8, _maskW, _maskH);

            byte[] outPng = await Task.Run(() => _inpainter!.ProcessImage(inputPng, maskPng));

            ImgOutput.Source = BytesToBitmap(outPng);
            SetStatus("Done.");
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.ToString(), "Inference failed", MessageBoxButton.OK, MessageBoxImage.Error);
            SetStatus("Inference failed.");
        }
        finally
        {
            SetBusy(false, null);
        }

        UpdateButtons();
    }

    private void BtnClearMask_Click(object sender, RoutedEventArgs e)
    {
        if (_inputBitmap == null) return;

        ClearMaskInternal();
        UpdateButtons();
        SetStatus("Mask cleared.");
    }

    // -----------------------------
    // Mode / brush UI
    // -----------------------------

    void MaskMode_Checked(object sender, RoutedEventArgs e)
    {
        // Checked 이벤트가 InitializeComponent() 중에도 발생할 수 있음
        if (!_uiReady || rbBrush == null || rbRect == null) return;

        _maskMode = (rbBrush.IsChecked == true) ? MaskMode.Brush : MaskMode.Rect;
        UpdateBrushUi();
    }

    void UpdateBrushUi()
    {
        if (!_uiReady) return;

        // 일부 컨트롤은 Loaded 이전/초기화 타이밍에 null일 수 있으므로 방어
        if (rbBrush == null || TxtBrushSizeLabel == null || SlBrushSize == null || TxtBrushSize == null)
            return;

        bool brush = (rbBrush.IsChecked == true);

        TxtBrushSizeLabel.Opacity = brush ? 1.0 : 0.4;
        SlBrushSize.IsEnabled = brush && !_busy;
        TxtBrushSize.IsEnabled = brush && !_busy;

        // sync text/slider -> radius
        int size = Clamp((int)Math.Round(SlBrushSize.Value), 4, 256);
        _brushRadiusPx = Math.Max(1, size / 2);
        if (TxtBrushSize.Text != size.ToString())
            TxtBrushSize.Text = size.ToString();

        UpdateBrushCursorVisual(lastMousePosition: null);
    }

    private void SlBrushSize_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (!_uiReady) return;

        int size = Clamp((int)Math.Round(SlBrushSize.Value), 4, 256);
        _brushRadiusPx = Math.Max(1, size / 2);

        if (TxtBrushSize.Text != size.ToString())
            TxtBrushSize.Text = size.ToString();

        UpdateBrushCursorVisual(lastMousePosition: null);
    }

    private void TxtBrushSize_TextChanged(object sender, System.Windows.Controls.TextChangedEventArgs e)
    {
        if (!_uiReady) return;

        if (int.TryParse(TxtBrushSize.Text, out int size))
        {
            size = Clamp(size, 4, 256);
            SlBrushSize.Value = size;
            _brushRadiusPx = Math.Max(1, size / 2);
        }

        UpdateBrushCursorVisual(lastMousePosition: null);
    }

    // -----------------------------
    // Overlay interaction
    // -----------------------------

    private void Overlay_MouseEnter(object sender, MouseEventArgs e)
    {
        _overlayHover = true;
        UpdateBrushCursorVisual(lastMousePosition: e.GetPosition(Overlay));
    }

    private void Overlay_MouseLeave(object sender, MouseEventArgs e)
    {
        _overlayHover = false;
        UpdateBrushCursorVisual(lastMousePosition: null);
    }

    private void Overlay_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (_busy || _inputBitmap == null) return;

        Overlay.CaptureMouse();

        if (_maskMode == MaskMode.Rect)
        {
            _dragging = true;
            _dragStart = e.GetPosition(Overlay);

            if (_rectVisual == null)
            {
                _rectVisual = new System.Windows.Shapes.Rectangle
                {
                    Stroke = Brushes.Cyan,
                    StrokeThickness = 2,
                    Fill = Brushes.Transparent,
                    StrokeDashArray = new DoubleCollection { 3, 2 }
                };
                Overlay.Children.Add(_rectVisual);
            }

            Canvas.SetLeft(_rectVisual, _dragStart.X);
            Canvas.SetTop(_rectVisual, _dragStart.Y);
            _rectVisual.Width = 0;
            _rectVisual.Height = 0;
        }
        else
        {
            _painting = true;
            var pt = e.GetPosition(Overlay);
            PaintBrushAt(pt);
        }
    }

    private void Overlay_MouseMove(object sender, MouseEventArgs e)
    {
        if (_inputBitmap == null) return;

        var pt = e.GetPosition(Overlay);

        if (_maskMode == MaskMode.Rect && _dragging && _rectVisual != null)
        {
            UpdateRectVisual(_dragStart, pt);
        }
        else if (_maskMode == MaskMode.Brush)
        {
            UpdateBrushCursorVisual(pt);

            if (_painting && e.LeftButton == MouseButtonState.Pressed && !_busy)
                PaintBrushAt(pt);
        }
    }

    private void Overlay_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        if (_busy || _inputBitmap == null) return;

        Overlay.ReleaseMouseCapture();

        if (_maskMode == MaskMode.Rect)
        {
            if (_dragging)
            {
                _dragging = false;

                var end = e.GetPosition(Overlay);
                ApplyRectMask(_dragStart, end);

                // keep rect visual but it is not needed anymore visually
                if (_rectVisual != null)
                {
                    // Optionally hide it after applying:
                    _rectVisual.Visibility = Visibility.Collapsed;
                }
            }
        }
        else
        {
            _painting = false;
        }

        UpdateButtons();
    }

    // -----------------------------
    // Rect mask
    // -----------------------------

    private void UpdateRectVisual(Point a, Point b)
    {
        if (_rectVisual == null) return;

        double x1 = Math.Min(a.X, b.X);
        double y1 = Math.Min(a.Y, b.Y);
        double x2 = Math.Max(a.X, b.X);
        double y2 = Math.Max(a.Y, b.Y);

        Canvas.SetLeft(_rectVisual, x1);
        Canvas.SetTop(_rectVisual, y1);
        _rectVisual.Width = Math.Max(0, x2 - x1);
        _rectVisual.Height = Math.Max(0, y2 - y1);
        _rectVisual.Visibility = Visibility.Visible;
    }

    private void ApplyRectMask(Point a, Point b)
    {
        if (_inputBitmap == null || _maskGray8 == null) return;

        var r = CanvasRectToImageRect(a, b);
        if (r.w <= 0 || r.h <= 0) return;

        for (int y = r.y; y < r.y + r.h; y++)
        {
            int row = y * _maskW;
            for (int x = r.x; x < r.x + r.w; x++)
                _maskGray8[row + x] = 255;
        }

        UpdateMaskPreview();
    }

    // -----------------------------
    // Brush mask
    // -----------------------------

    private void PaintBrushAt(Point ptCanvas)
    {
        if (_inputBitmap == null || _maskGray8 == null) return;

        // canvas pt -> image px
        var p = CanvasPointToImagePoint(ptCanvas);
        if (p.x < 0 || p.y < 0 || p.x >= _maskW || p.y >= _maskH) return;

        int cx = p.x;
        int cy = p.y;
        int r = _brushRadiusPx;
        int r2 = r * r;

        int y0 = Math.Max(0, cy - r);
        int y1 = Math.Min(_maskH - 1, cy + r);
        int x0 = Math.Max(0, cx - r);
        int x1 = Math.Min(_maskW - 1, cx + r);

        for (int y = y0; y <= y1; y++)
        {
            int dy = y - cy;
            int dy2 = dy * dy;
            int row = y * _maskW;

            for (int x = x0; x <= x1; x++)
            {
                int dx = x - cx;
                if (dx * dx + dy2 <= r2)
                    _maskGray8[row + x] = 255;
            }
        }

        UpdateMaskPreview();
    }

    private void UpdateBrushCursorVisual(Point? lastMousePosition)
    {
        if (!_overlayHover || _busy || _inputBitmap == null || rbBrush.IsChecked != true)
        {
            BrushCursor.Visibility = Visibility.Collapsed;
            return;
        }

        Point pt = lastMousePosition ?? Mouse.GetPosition(Overlay);

        double cw = Overlay.ActualWidth;
        double ch = Overlay.ActualHeight;
        if (cw <= 2 || ch <= 2) { BrushCursor.Visibility = Visibility.Collapsed; return; }

        int iw = _inputBitmap.PixelWidth;
        int ih = _inputBitmap.PixelHeight;

        // Determine displayed image rect within the overlay (Uniform stretch)
        double imgAspect = (double)iw / ih;
        double canvasAspect = cw / ch;

        double dispW, dispH, offX, offY;
        if (canvasAspect > imgAspect)
        {
            dispH = ch;
            dispW = ch * imgAspect;
            offX = (cw - dispW) / 2;
            offY = 0;
        }
        else
        {
            dispW = cw;
            dispH = cw / imgAspect;
            offX = 0;
            offY = (ch - dispH) / 2;
        }

        // If mouse is outside displayed image region, hide cursor.
        if (pt.X < offX || pt.X > offX + dispW || pt.Y < offY || pt.Y > offY + dispH)
        {
            BrushCursor.Visibility = Visibility.Collapsed;
            return;
        }

        double scale = dispW / iw; // == dispH / ih
        double rDisp = Math.Max(2.0, _brushRadiusPx * scale);

        BrushCursor.Width = rDisp * 2;
        BrushCursor.Height = rDisp * 2;
        Canvas.SetLeft(BrushCursor, pt.X - rDisp);
        Canvas.SetTop(BrushCursor, pt.Y - rDisp);
        BrushCursor.Visibility = Visibility.Visible;
    }

    // -----------------------------
    // Mask + preview
    // -----------------------------

    private void EnsureMaskBufferForInput()
    {
        if (_inputBitmap == null) return;

        _maskW = _inputBitmap.PixelWidth;
        _maskH = _inputBitmap.PixelHeight;
        _maskGray8 = new byte[_maskW * _maskH];
    }

    private void ClearMaskInternal()
    {
        if (_maskGray8 == null) return;
        Array.Clear(_maskGray8, 0, _maskGray8.Length);

        if (_rectVisual != null)
            _rectVisual.Visibility = Visibility.Collapsed;

        UpdateMaskPreview();
        UpdateBrushCursorVisual(lastMousePosition: null);
    }

    private bool HasAnyMask()
    {
        if (_maskGray8 == null) return false;
        for (int i = 0; i < _maskGray8.Length; i++)
            if (_maskGray8[i] != 0) return true;
        return false;
    }

    private void UpdateMaskPreview()
    {
        if (_maskGray8 == null || _maskW <= 0 || _maskH <= 0)
        {
            ImgMaskPreview.Source = null;
            return;
        }

        // simple red overlay: alpha from mask
        // create BGRA32 image same size
        int stride = _maskW * 4;
        byte[] bgra = new byte[_maskW * _maskH * 4];

        for (int i = 0; i < _maskGray8.Length; i++)
        {
            byte a = _maskGray8[i];
            int j = i * 4;
            bgra[j + 0] = 0;   // B
            bgra[j + 1] = 0;   // G
            bgra[j + 2] = 255; // R
            bgra[j + 3] = a;   // A
        }

        var bmp = BitmapSource.Create(_maskW, _maskH, 96, 96, PixelFormats.Bgra32, null, bgra, stride);
        bmp.Freeze();
        ImgMaskPreview.Source = bmp;
    }

    // -----------------------------
    // Coordinate mapping (Uniform stretch)
    // -----------------------------

    private (int x, int y) CanvasPointToImagePoint(Point ptCanvas)
    {
        if (_inputBitmap == null) return (-1, -1);

        double cw = Overlay.ActualWidth;
        double ch = Overlay.ActualHeight;
        if (cw <= 2 || ch <= 2) return (-1, -1);

        int iw = _inputBitmap.PixelWidth;
        int ih = _inputBitmap.PixelHeight;

        double imgAspect = (double)iw / ih;
        double canvasAspect = cw / ch;

        double dispW, dispH, offX, offY;
        if (canvasAspect > imgAspect)
        {
            dispH = ch;
            dispW = ch * imgAspect;
            offX = (cw - dispW) / 2;
            offY = 0;
        }
        else
        {
            dispW = cw;
            dispH = cw / imgAspect;
            offX = 0;
            offY = (ch - dispH) / 2;
        }

        double x = (ptCanvas.X - offX) / dispW;
        double y = (ptCanvas.Y - offY) / dispH;

        int px = (int)Math.Round(x * (iw - 1));
        int py = (int)Math.Round(y * (ih - 1));
        return (px, py);
    }

    private (int x, int y, int w, int h) CanvasRectToImageRect(Point a, Point b)
    {
        var p1 = CanvasPointToImagePoint(a);
        var p2 = CanvasPointToImagePoint(b);

        int x1 = Math.Min(p1.x, p2.x);
        int y1 = Math.Min(p1.y, p2.y);
        int x2 = Math.Max(p1.x, p2.x);
        int y2 = Math.Max(p1.y, p2.y);

        x1 = Clamp(x1, 0, _maskW - 1);
        x2 = Clamp(x2, 0, _maskW - 1);
        y1 = Clamp(y1, 0, _maskH - 1);
        y2 = Clamp(y2, 0, _maskH - 1);

        int w = Math.Max(0, x2 - x1);
        int h = Math.Max(0, y2 - y1);
        return (x1, y1, w, h);
    }

    // -----------------------------
    // UI / state
    // -----------------------------

    private void SetBusy(bool busy, string? status)
    {
        _busy = busy;
        if (status != null) SetStatus(status);

        PbarLoading.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;

        BtnPickModel.IsEnabled = !busy;
        BtnOpenImage.IsEnabled = !busy;
        BtnRun.IsEnabled = !busy && _inpainter != null && _inputBitmap != null && HasAnyMask();
        BtnClearMask.IsEnabled = !busy && _inputBitmap != null && HasAnyMask();

        rbRect.IsEnabled = !busy;
        rbBrush.IsEnabled = !busy;
        SlBrushSize.IsEnabled = !busy && rbBrush.IsChecked == true;
        TxtBrushSize.IsEnabled = !busy && rbBrush.IsChecked == true;
    }

    private void UpdateButtons()
    {
        BtnRun.IsEnabled = !_busy && _inpainter != null && _inputBitmap != null && HasAnyMask();
        BtnClearMask.IsEnabled = !_busy && _inputBitmap != null && HasAnyMask();
    }

    private void SetStatus(string text)
    {
        TxtStatus.Text = text;
    }

    // -----------------------------
    // PNG helpers
    // -----------------------------

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

    private static byte[] BitmapToPngBytes(BitmapSource bmp)
    {
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(bmp));
        using var ms = new MemoryStream();
        encoder.Save(ms);
        return ms.ToArray();
    }

    private static byte[] Gray8MaskToPngBytes(byte[] gray8, int w, int h)
    {
        // Gray8 PNG
        var bmp = BitmapSource.Create(w, h, 96, 96, PixelFormats.Gray8, null, gray8, w);
        bmp.Freeze();
        return BitmapToPngBytes(bmp);
    }

    private static int Clamp(int v, int min, int max) => (v < min) ? min : (v > max ? max : v);
}

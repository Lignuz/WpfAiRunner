using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using LamaEngine;

namespace LamaWpf;

public partial class MainWindow : Window
{
    private LamaInpainter? _inpainter;
    private string? _modelPath;
    private BitmapSource? _inputBitmap;

    private Point _dragStart;
    private bool _dragging;
    private System.Windows.Shapes.Rectangle? _rectVisual;
    private Int32Rect? _selectedRectPx;

    private bool _busy;

    public MainWindow()
    {
        InitializeComponent();

        TxtModel.Text = "(no model)";
        SetStatus("Ready.");

        UpdateRunEnabled();
    }

    protected override void OnClosed(EventArgs e)
    {
        _inpainter?.Dispose();
        _inpainter = null;
        base.OnClosed(e);
    }

    private void SetStatus(string message)
    {
        // 너무 길면 UI가 보기 싫어서 컷 
        if (message.Length > 300) message = message[..300] + "…";
        TxtStatus.Text = message;
    }

    private void SetBusy(bool busy, string? status = null)
    {
        _busy = busy;

        if (status != null) SetStatus(status);

        PbarLoading.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;

        // 작업 중에는 전부 비활성화
        BtnPickModel.IsEnabled = !busy;
        BtnOpenImage.IsEnabled = !busy;
        BtnRun.IsEnabled = false; // 아래 UpdateRunEnabled에서 idle일 때만 다시 켜짐
        Overlay.IsEnabled = !busy;

        Mouse.OverrideCursor = busy ? Cursors.Wait : null;

        UpdateRunEnabled();
    }

    private async void BtnPickModel_Click(object sender, RoutedEventArgs e)
    {
        if (_busy) return;

        var dlg = new OpenFileDialog
        {
            Title = "Select LaMa ONNX model",
            Filter = "ONNX model|*.onnx|All files (*.*)|*.*"
        };
        if (dlg.ShowDialog() != true) return;

        string selectedPath = dlg.FileName;
        if (!File.Exists(selectedPath))
        {
            MessageBox.Show("Model file not found.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        SetBusy(true, "Loading model...");

        try
        {
            // 백그라운드에서 로딩
            LamaInpainter loadedInpainter = await Task.Run(() =>
            {
                return new LamaInpainter(selectedPath);
            });

            _inpainter?.Dispose();
            _inpainter = loadedInpainter;
            _modelPath = selectedPath;

            TxtModel.Text = Path.GetFileName(_modelPath);
            SetStatus("Model loaded.");
        }
        catch (Exception ex)
        {
            SetStatus("Load failed.");
            MessageBox.Show($"Failed to load model:\n{ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void BtnOpenImage_Click(object sender, RoutedEventArgs e)
    {
        if (_busy) return;

        var dlg = new OpenFileDialog
        {
            Title = "Open Image",
            Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp;*.webp|All files (*.*)|*.*"
        };
        if (dlg.ShowDialog() != true) return;

        try
        {
            LoadInput(dlg.FileName);
            ClearSelection();
            ImgOutput.Source = null;
            SetStatus("Image loaded.");
        }
        catch (Exception ex)
        {
            SetStatus("Image load failed.");
            MessageBox.Show($"Failed to load image:\n{ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }

        UpdateRunEnabled();
    }

    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        if (_busy) return;
        if (_inpainter == null || _inputBitmap == null || _selectedRectPx == null) return;

        var r = _selectedRectPx.Value;
        if (r.Width < 2 || r.Height < 2)
        {
            MessageBox.Show("Mask rectangle is too small.", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        SetBusy(true, "Processing...");

        try
        {
            byte[] inputBytes = BitmapToBytes(_inputBitmap);
            byte[] maskBytes = CreateMaskBytes(_inputBitmap.PixelWidth, _inputBitmap.PixelHeight, r);

            // 백그라운드에서 처리
            byte[] resultBytes = await Task.Run(() =>
            {
                return _inpainter.ProcessImage(inputBytes, maskBytes);
            });

            ImgOutput.Source = BytesToBitmap(resultBytes);
            SetStatus("Done.");
        }
        catch (Exception ex)
        {
            SetStatus("Error.");
            MessageBox.Show($"Processing Error:\n{ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void UpdateRunEnabled()
    {
        BtnRun.IsEnabled = !_busy && _inpainter != null && _inputBitmap != null && _selectedRectPx != null;
    }

    private void LoadInput(string path)
    {
        var bmp = new BitmapImage();
        bmp.BeginInit();
        bmp.CacheOption = BitmapCacheOption.OnLoad;
        bmp.UriSource = new Uri(path);
        bmp.EndInit();
        bmp.Freeze();

        _inputBitmap = bmp;
        ImgInput.Source = _inputBitmap;
    }

    private byte[] BitmapToBytes(BitmapSource bmp)
    {
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(bmp));
        using var ms = new MemoryStream();
        encoder.Save(ms);
        return ms.ToArray();
    }

    private BitmapImage BytesToBitmap(byte[] bytes)
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

    private byte[] CreateMaskBytes(int width, int height, Int32Rect rect)
    {
        // Gray8: 0=background, 255=mask
        var wb = new WriteableBitmap(width, height, 96, 96, PixelFormats.Gray8, null);
        int stride = width;
        byte[] pixels = new byte[stride * height];

        int x0 = Math.Clamp(rect.X, 0, width - 1);
        int y0 = Math.Clamp(rect.Y, 0, height - 1);
        int x1 = Math.Clamp(rect.X + rect.Width, x0 + 1, width);
        int y1 = Math.Clamp(rect.Y + rect.Height, y0 + 1, height);

        for (int y = y0; y < y1; y++)
        {
            int row = y * stride;
            for (int x = x0; x < x1; x++) pixels[row + x] = 255;
        }

        wb.WritePixels(new Int32Rect(0, 0, width, height), pixels, stride, 0);

        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(wb));
        using var ms = new MemoryStream();
        encoder.Save(ms);
        return ms.ToArray();
    }

    // --- Overlay & Selection Logic ---
    private void Overlay_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (_inputBitmap == null || _busy) return;

        _dragging = true;
        _dragStart = e.GetPosition(Overlay);
        Overlay.CaptureMouse();

        if (_rectVisual == null)
        {
            _rectVisual = new System.Windows.Shapes.Rectangle
            {
                Stroke = Brushes.DeepSkyBlue,
                StrokeThickness = 2,
                Fill = new SolidColorBrush(Color.FromArgb(40, 0, 191, 255))
            };
            Overlay.Children.Add(_rectVisual);
        }

        Canvas.SetLeft(_rectVisual, _dragStart.X);
        Canvas.SetTop(_rectVisual, _dragStart.Y);
        _rectVisual.Width = 0;
        _rectVisual.Height = 0;
    }

    private void Overlay_MouseMove(object sender, MouseEventArgs e)
    {
        if (!_dragging || _rectVisual == null) return;

        var p = e.GetPosition(Overlay);
        var x = Math.Min(p.X, _dragStart.X);
        var y = Math.Min(p.Y, _dragStart.Y);
        var w = Math.Abs(p.X - _dragStart.X);
        var h = Math.Abs(p.Y - _dragStart.Y);

        Canvas.SetLeft(_rectVisual, x);
        Canvas.SetTop(_rectVisual, y);
        _rectVisual.Width = w;
        _rectVisual.Height = h;
    }

    private void Overlay_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        if (!_dragging || _rectVisual == null || _inputBitmap == null) return;

        _dragging = false;
        Overlay.ReleaseMouseCapture();

        var overlayRect = new Rect(
            Canvas.GetLeft(_rectVisual),
            Canvas.GetTop(_rectVisual),
            _rectVisual.Width,
            _rectVisual.Height);

        _selectedRectPx = OverlayRectToPixelRect(overlayRect);

        UpdateRunEnabled();
        SetStatus($"Mask selected: {_selectedRectPx.Value.Width}x{_selectedRectPx.Value.Height}px");
    }

    private void ClearSelection()
    {
        _selectedRectPx = null;

        if (_rectVisual != null)
        {
            Overlay.Children.Remove(_rectVisual);
            _rectVisual = null;
        }

        UpdateRunEnabled();
    }

    private Int32Rect OverlayRectToPixelRect(Rect overlayRect)
    {
        if (_inputBitmap == null) return new Int32Rect(0, 0, 0, 0);

        double iw = _inputBitmap.PixelWidth;
        double ih = _inputBitmap.PixelHeight;

        double cw = Overlay.ActualWidth;
        double ch = Overlay.ActualHeight;

        if (cw <= 0 || ch <= 0 || iw <= 0 || ih <= 0)
            return new Int32Rect(0, 0, 0, 0);

        double scale = Math.Min(cw / iw, ch / ih);
        double dispW = iw * scale;
        double dispH = ih * scale;
        double offX = (cw - dispW) / 2.0;
        double offY = (ch - dispH) / 2.0;

        double x1 = Math.Clamp(overlayRect.X, offX, offX + dispW);
        double y1 = Math.Clamp(overlayRect.Y, offY, offY + dispH);
        double x2 = Math.Clamp(overlayRect.X + overlayRect.Width, offX, offX + dispW);
        double y2 = Math.Clamp(overlayRect.Y + overlayRect.Height, offY, offY + dispH);

        int px1 = (int)((x1 - offX) / scale);
        int py1 = (int)((y1 - offY) / scale);
        int px2 = (int)((x2 - offX) / scale);
        int py2 = (int)((y2 - offY) / scale);

        return new Int32Rect(px1, py1, Math.Max(0, px2 - px1), Math.Max(0, py2 - py1));
    }
}

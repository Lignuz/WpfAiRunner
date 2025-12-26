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
    private string? _imagePath;
    private BitmapSource? _inputBitmap;

    private Point _dragStart;
    private bool _dragging;
    private System.Windows.Shapes.Rectangle? _rectVisual;
    private Int32Rect? _selectedRectPx;

    public MainWindow()
    {
        InitializeComponent();
        _modelPath = null;
        TxtModel.Text = "(no model)";
        UpdateRunEnabled();
    }

    private async void BtnPickModel_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title = "Select ONNX Model",
            Filter = "ONNX model (*.onnx)|*.onnx|All files (*.*)|*.*"
        };

        if (dlg.ShowDialog() != true) return;

        string selectedPath = dlg.FileName;

        // 1. UI를 '로딩 중' 상태로 변경
        TxtStatus.Text = "Loading model... Please wait.";
        PbarLoading.Visibility = Visibility.Visible; // 프로그래스바 표시
        BtnPickModel.IsEnabled = false; // 중복 클릭 방지
        Mouse.OverrideCursor = Cursors.Wait; // 마우스 커서를 모래시계로

        try
        {
            // 2. 백그라운드 스레드에서 무거운 작업(모델 로딩) 실행
            // UI 스레드는 여기서 멈추지 않고 계속 반응(창 이동 등)할 수 있음
            LamaInpainter loadedInpainter = await Task.Run(() =>
            {
                // 이 블록 안은 백그라운드에서 돕니다.
                return new LamaInpainter(selectedPath);
            });

            // 3. 로딩 성공 시 UI 업데이트 (다시 메인 스레드)
            // 기존 모델이 있다면 정리
            _inpainter?.Dispose();

            _inpainter = loadedInpainter;
            _modelPath = selectedPath;

            TxtModel.Text = Path.GetFileName(_modelPath); // 파일명만 보여주기 (깔끔하게)
            TxtStatus.Text = "Model loaded successfully.";
        }
        catch (Exception ex)
        {
            // 4. 실패 처리
            TxtStatus.Text = "Load failed.";
            MessageBox.Show($"Failed to load model:\n{ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            // 5. UI 상태 복구 (성공/실패 여부와 상관없이 실행)
            Mouse.OverrideCursor = null;
            BtnPickModel.IsEnabled = true;
            PbarLoading.Visibility = Visibility.Collapsed; // 프로그래스바 숨김
            UpdateRunEnabled();
        }
    }

    private void BtnOpenImage_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title = "Open Image",
            Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp;*.webp|All files (*.*)|*.*"
        };
        if (dlg.ShowDialog() != true) return;

        _imagePath = dlg.FileName;
        LoadInput(_imagePath);
        ClearSelection();
        ImgOutput.Source = null;
        TxtStatus.Text = "Image loaded.";
        UpdateRunEnabled();
    }

    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        if (_inpainter == null || _inputBitmap == null || _selectedRectPx == null) return;

        try
        {
            TxtStatus.Text = "Processing...";
            BtnRun.IsEnabled = false;
            Mouse.OverrideCursor = Cursors.Wait;

            byte[] inputBytes = BitmapToBytes(_inputBitmap);
            byte[] maskBytes = CreateMaskBytes(_inputBitmap.PixelWidth, _inputBitmap.PixelHeight, _selectedRectPx.Value);

            // 비동기 실행 (UI 멈춤 방지)
            byte[] resultBytes = await Task.Run(() => 
            {
                return _inpainter.ProcessImage(inputBytes, maskBytes);
            });

            ImgOutput.Source = BytesToBitmap(resultBytes);
            TxtStatus.Text = "Done.";
        }
        catch (Exception ex)
        {
            TxtStatus.Text = "Error.";
            MessageBox.Show($"Processing Error: {ex.Message}");
        }
        finally
        {
            Mouse.OverrideCursor = null;
            UpdateRunEnabled();
        }
    }

    private void UpdateRunEnabled()
    {
        BtnRun.IsEnabled = _inpainter != null && _imagePath != null && _selectedRectPx != null;
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
        if (_inputBitmap == null) return;
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
        _selectedRectPx = null;
        UpdateRunEnabled();
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
        if (!_dragging) return;
        _dragging = false;
        Overlay.ReleaseMouseCapture();

        if (_rectVisual == null || _inputBitmap == null) return;
        
        var overlayRect = new Int32Rect(
            (int)Canvas.GetLeft(_rectVisual), (int)Canvas.GetTop(_rectVisual),
            (int)_rectVisual.Width, (int)_rectVisual.Height);

        var px = MapOverlayRectToImagePixels(overlayRect, _inputBitmap);
        if (px.Width < 2 || px.Height < 2) { ClearSelection(); return; }

        _selectedRectPx = px;
        TxtStatus.Text = $"Selected: {px.Width}x{px.Height}";
        UpdateRunEnabled();
    }

    private void ClearSelection()
    {
        _selectedRectPx = null;
        if (_rectVisual != null) { _rectVisual.Width = 0; _rectVisual.Height = 0; }
        UpdateRunEnabled();
    }

    private Int32Rect MapOverlayRectToImagePixels(Int32Rect overlayRect, BitmapSource img)
    {
        double cw = Overlay.ActualWidth;
        double ch = Overlay.ActualHeight;
        double iw = img.PixelWidth;
        double ih = img.PixelHeight;

        double s = Math.Min(cw / iw, ch / ih);
        double dispW = iw * s;
        double dispH = ih * s;
        double offX = (cw - dispW) / 2.0;
        double offY = (ch - dispH) / 2.0;

        double x1 = Math.Clamp(overlayRect.X, offX, offX + dispW);
        double y1 = Math.Clamp(overlayRect.Y, offY, offY + dispH);
        double x2 = Math.Clamp(overlayRect.X + overlayRect.Width, offX, offX + dispW);
        double y2 = Math.Clamp(overlayRect.Y + overlayRect.Height, offY, offY + dispH);

        int px1 = (int)((x1 - offX) / s);
        int py1 = (int)((y1 - offY) / s);
        int px2 = (int)((x2 - offX) / s);
        int py2 = (int)((y2 - offY) / s);

        return new Int32Rect(px1, py1, Math.Max(0, px2 - px1), Math.Max(0, py2 - py1));
    }
}
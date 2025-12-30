using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using OnnxEngines.Depth;

namespace WpfAiRunner.Views;

public partial class DepthView : UserControl, IDisposable
{
    private DepthEstimator? _estimator;
    private BitmapSource? _inputBitmap;
    private string? _modelPath; // 현재 모델 경로 기억
    private bool _busy;

    public DepthView() => InitializeComponent();

    public void Dispose() => _estimator?.Dispose();

    private void UserControl_Loaded(object sender, RoutedEventArgs e)
    {
        UpdateButtons();
    }

    // 1. 모델 선택 버튼
    private async void BtnPickModel_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "ONNX (*.onnx)|*.onnx" };
        if (dlg.ShowDialog(Window.GetWindow(this)) != true) return;

        // 선택한 경로로 로딩 (공통 함수 호출)
        await ReloadModelAsync(dlg.FileName);
    }

    // 2. GPU 체크박스 클릭 (추가됨)
    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e)
    {
        // 이미 모델이 로드된 상태라면, 현재 경로로 다시 로딩
        if (!string.IsNullOrEmpty(_modelPath))
        {
            await ReloadModelAsync(_modelPath);
        }
    }

    // 3. 모델 로딩 공통 로직 (리팩토링)
    private async Task ReloadModelAsync(string path)
    {
        SetBusy(true);
        TxtStatus.Text = "Loading model...";

        // UI 갱신 대기
        await Task.Delay(10);

        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;

            // 기존 것 정리
            _estimator?.Dispose();
            _estimator = null;

            // 백그라운드 생성
            _estimator = await Task.Run(() => new DepthEstimator(path, useGpu));

            _modelPath = path;

            // UI 업데이트: 모델 이름 + 현재 모드(CPU/GPU) 표시
            TxtModel.Text = Path.GetFileName(path);
            TxtStatus.Text = $"Loaded on {_estimator.DeviceMode}";

            // GPU 실패 시 CPU로 자동 전환된 경우 처리
            if (useGpu && _estimator.DeviceMode.Contains("CPU"))
            {
                ChkUseGpu.IsChecked = false;
                MessageBox.Show("GPU init failed. Fallback to CPU.", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Load failed: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            TxtStatus.Text = "Load failed.";
            TxtModel.Text = "(error)";
        }
        finally
        {
            SetBusy(false);
            // 로딩 중 발생한 임시 메모리 정리
            GC.Collect();
        }
    }

    private void BtnOpenImage_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp" };
        if (dlg.ShowDialog(Window.GetWindow(this)) != true) return;

        var bmp = new BitmapImage(new Uri(dlg.FileName));
        bmp.Freeze();
        _inputBitmap = bmp;
        ImgInput.Source = bmp;
        ImgOutput.Source = null;

        TxtStatus.Text = "Image loaded.";
        UpdateButtons();
    }

    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        if (_estimator == null || _inputBitmap == null) return;

        SetBusy(true);
        TxtStatus.Text = "Estimating...";

        try
        {
            byte[] inputBytes = BitmapToBytes(_inputBitmap);

            byte[] outBytes = await Task.Run(() => _estimator.ProcessImage(inputBytes));

            ImgOutput.Source = BytesToBitmap(outBytes);
            TxtStatus.Text = "Done.";
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error: {ex.Message}");
            TxtStatus.Text = "Failed.";
        }
        finally
        {
            SetBusy(false);
            GC.Collect();
        }
    }

    private void SetBusy(bool busy)
    {
        _busy = busy;
        PbarLoading.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;

        // 로딩 중에는 체크박스 못 건드리게 잠금 (중복 실행 방지)
        ChkUseGpu.IsEnabled = !busy;

        UpdateButtons();
    }

    private void UpdateButtons()
    {
        BtnPickModel.IsEnabled = !_busy;
        BtnOpenImage.IsEnabled = !_busy;
        BtnRun.IsEnabled = !_busy && _estimator != null && _inputBitmap != null;
    }

    private static byte[] BitmapToBytes(BitmapSource bmp)
    {
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(bmp));
        using var ms = new MemoryStream();
        encoder.Save(ms);
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
}
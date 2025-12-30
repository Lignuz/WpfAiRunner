using OnnxEngines.Upscaling;
using Microsoft.Win32;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace WpfAiRunner.Views;

public partial class RealEsrganView : UserControl, IDisposable
{
    private readonly RealEsrganEngine _engine = new();
    private byte[]? _inputBytes;

    // 현재 로드된 모델 경로 저장 (재로딩용)
    private string? _currentModelPath;

    public RealEsrganView() => InitializeComponent();
    public void Dispose() => _engine.Dispose();

    // 1. 모델 로드 버튼
    private async void BtnLoadModel_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "ONNX Model|*.onnx" };
        if (dlg.ShowDialog() != true) return;

        _currentModelPath = dlg.FileName;
        await ReloadModel();
    }

    // 2. GPU 체크박스 토글 (자동 재로딩)
    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e)
    {
        // 모델이 로드된 적이 없다면 아무것도 안 함
        if (string.IsNullOrEmpty(_currentModelPath)) return;

        // 모델 다시 읽기
        await ReloadModel();
    }

    // [공통] 모델 로드 로직
    private async Task ReloadModel()
    {
        if (string.IsNullOrEmpty(_currentModelPath)) return;

        SetBusy(true, "Loading Model...");
        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;

            // UI 스레드 멈춤 방지를 위해 Task.Run 사용
            await Task.Run(() => _engine.LoadModel(_currentModelPath, useGpu));

            TxtStatus.Text = $"Model Loaded ({_engine.DeviceMode})";
            BtnOpenImage.IsEnabled = true;

            // 만약 이미지가 열려있다면 업스케일 버튼 활성화
            if (_inputBytes != null) BtnUpscale.IsEnabled = true;
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error loading model: {ex.Message}");
            TxtStatus.Text = "Load Failed";
            _currentModelPath = null; // 실패 시 경로 초기화
        }
        finally
        {
            SetBusy(false);
        }
    }

    // 3. 이미지 열기
    private void BtnOpenImage_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "Images|*.png;*.jpg;*.jpeg" };
        if (dlg.ShowDialog() != true) return;

        var bmp = new BitmapImage(new Uri(dlg.FileName));
        ImgInput.Source = bmp;

        using var ms = new MemoryStream();
        var enc = new PngBitmapEncoder();
        enc.Frames.Add(BitmapFrame.Create(bmp));
        enc.Save(ms);
        _inputBytes = ms.ToArray();

        BtnUpscale.IsEnabled = true;
        BtnSave.IsEnabled = false; // 새 이미지를 열면 저장 버튼 비활성
        ImgOutput.Source = null;
        TxtStatus.Text = "Image Loaded.";
    }

    // 4. 업스케일 실행
    private async void BtnUpscale_Click(object sender, RoutedEventArgs e)
    {
        if (_inputBytes == null) return;

        SetBusy(true, "Upscaling... 0%");
        PbarUpscale.Visibility = Visibility.Visible;
        PbarUpscale.Value = 0;

        var progress = new Progress<double>(p =>
        {
            PbarUpscale.Value = p * 100;
            TxtStatus.Text = $"Upscaling... {(int)(p * 100)}%";
        });

        try
        {
            byte[] resultBytes = await Task.Run(() => _engine.Upscale(_inputBytes, progress));

            using var ms = new MemoryStream(resultBytes);
            var resultBmp = new BitmapImage();
            resultBmp.BeginInit();
            resultBmp.StreamSource = ms;
            resultBmp.CacheOption = BitmapCacheOption.OnLoad;
            resultBmp.EndInit();
            resultBmp.Freeze();

            ImgOutput.Source = resultBmp;
            BtnSave.IsEnabled = true; // 완료 후 저장 버튼 활성화
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
            PbarUpscale.Visibility = Visibility.Hidden;
        }
    }

    // 5. 저장 기능
    private void BtnSave_Click(object sender, RoutedEventArgs e)
    {
        if (ImgOutput.Source is not BitmapSource resultBmp) return;

        var dlg = new SaveFileDialog { Filter = "PNG Image|*.png", FileName = "upscaled.png" };
        if (dlg.ShowDialog() != true) return;

        try
        {
            using var fileStream = new FileStream(dlg.FileName, FileMode.Create);
            var encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(resultBmp));
            encoder.Save(fileStream);
            MessageBox.Show("Image saved successfully!", "Success", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Failed to save: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    // 6. UI 잠금 헬퍼
    private void SetBusy(bool busy, string? statusMsg = null)
    {
        // 작업 중에는 모든 입력 컨트롤 비활성화
        BtnLoadModel.IsEnabled = !busy;
        ChkUseGpu.IsEnabled = !busy;
        BtnOpenImage.IsEnabled = !busy && !string.IsNullOrEmpty(_currentModelPath); // 모델 없으면 안 켜짐
        BtnUpscale.IsEnabled = !busy && _inputBytes != null && !string.IsNullOrEmpty(_currentModelPath);
        BtnSave.IsEnabled = !busy && ImgOutput.Source != null;

        if (statusMsg != null)
        {
            TxtStatus.Text = statusMsg;
        }
    }
}
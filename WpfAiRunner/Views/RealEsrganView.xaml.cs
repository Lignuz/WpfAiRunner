using OnnxEngines.Upscaling;
using OnnxEngines.Utils;
using System.Windows;
using System.Windows.Controls;

namespace WpfAiRunner.Views;

public partial class RealEsrganView : BaseAiView 
{
    private readonly RealEsrganEngine _engine = new();
    private string? _currentModelPath;

    // 부모에게 UI 컨트롤 연결
    protected override Image ControlImgInput => ImgInput;
    protected override Image? ControlImgOutput => ImgOutput;
    protected override ProgressBar? ControlPbarLoading => PbarStatus; // 이름이 PbarStatus임에 주의
    protected override TextBlock? ControlTxtStatus => TxtStatus;

    public RealEsrganView()
    {
        InitializeComponent();
    }

    public override void Dispose() => _engine.Dispose();

    // 부모의 Loaded 이후 실행됨
    protected override async void OnLoaded(RoutedEventArgs e)
    {
#if DEBUG
        if (string.IsNullOrEmpty(_currentModelPath))
        {
            string? debugPath = OnnxHelper.FindModelInDebug("Real-ESRGAN-x4plus.onnx");
            if (debugPath != null)
            {
                _currentModelPath = debugPath;
                await ReloadModel();
            }
        }
#endif
        UpdateUi();
    }

    // 이미지가 로드될 때마다 호출됨
    protected override void OnImageLoaded()
    {
        UpdateUi();
    }

    private async void BtnLoadModel_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new Microsoft.Win32.OpenFileDialog { Filter = "ONNX Model|*.onnx" };
        if (dlg.ShowDialog() != true) return;

        _currentModelPath = dlg.FileName;
        await ReloadModel();
    }

    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(_currentModelPath)) return;
        await ReloadModel();
    }

    private async Task ReloadModel()
    {
        if (string.IsNullOrEmpty(_currentModelPath)) return;

        SetBusyState(true);
        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;
            await Task.Run(() => _engine.LoadModel(_currentModelPath, useGpu));

            Log($"Model Loaded ({_engine.DeviceMode})");
            if (useGpu && _engine.DeviceMode.Contains("CPU"))
            {
                ChkUseGpu.IsChecked = false;
            }
            UpdateUi();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error loading model: {ex.Message}");
            Log("Load Failed");
            _currentModelPath = null;
        }
        finally
        {
            SetBusyState(false);
        }
    }

    // 부모 메서드 호출
    private void BtnOpenImage_Click(object sender, RoutedEventArgs e) => OpenImageDialog();

    // 부모 메서드 호출
    private void BtnSave_Click(object sender, RoutedEventArgs e) => SaveOutputImage("upscaled.png");

    private async void BtnUpscale_Click(object sender, RoutedEventArgs e)
    {
        if (_inputBitmap == null) return;

        SetBusyState(true);
        // 업스케일링은 진행률 표시가 필요하므로 Indeterminate 끄기
        if (ControlPbarLoading != null)
        {
            ControlPbarLoading.IsIndeterminate = false;
            ControlPbarLoading.Value = 0;
        }

        var progress = new Progress<double>(p =>
        {
            if (ControlPbarLoading != null) ControlPbarLoading.Value = p * 100;
            Log($"Upscaling... {(int)(p * 100)}%");
        });

        try
        {
            // 부모의 헬퍼를 사용해 바이트 변환
            byte[] inputBytes = BitmapToBytes(_inputBitmap);
            byte[] resultBytes = await Task.Run(() => _engine.Upscale(inputBytes, progress));

            ImgOutput.Source = BytesToBitmap(resultBytes);
            Log("Done.");
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error: {ex.Message}");
            Log("Failed.");
        }
        finally
        {
            SetBusyState(false);
        }
    }

    private void UpdateUi()
    {
        BtnOpenImage.IsEnabled = true;
        BtnUpscale.IsEnabled = _inputBitmap != null && !string.IsNullOrEmpty(_currentModelPath);
        BtnSave.IsEnabled = ImgOutput.Source != null;
    }
}
using Microsoft.Win32;
using OnnxEngines.Colorization;
using OnnxEngines.Utils;
using System.Windows;
using System.Windows.Controls;

namespace WpfAiRunner.Views;

public partial class ColorizationView : BaseAiView
{
    private ColorizationEngine? _engine;
    private string? _currentModelPath;

    protected override Image ControlImgInput => ImgInput;
    protected override Image? ControlImgOutput => ImgOutput;
    protected override ProgressBar? ControlPbarLoading => PbarLoading;
    protected override TextBlock? ControlTxtStatus => TxtStatus;

    public ColorizationView() => InitializeComponent();
    public override void Dispose() => _engine?.Dispose();

    protected override async void OnLoaded(RoutedEventArgs e)
    {
#if DEBUG
        if (_engine == null)
        {
            string? debugPath = OnnxHelper.FindModelInDebug("ddcolor.onnx");
            if (debugPath != null)
            {
                _currentModelPath = debugPath;
                await ReloadModel();
            }
        }
#endif
        UpdateUi();
    }

    protected override void OnImageLoaded() => UpdateUi();

    private async void BtnLoadModel_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "ONNX (*.onnx)|*.onnx" };
        if (dlg.ShowDialog() == true)
        {
            _currentModelPath = dlg.FileName;
            await ReloadModel();
        }
    }

    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e)
    {
        if (!string.IsNullOrEmpty(_currentModelPath))
            await ReloadModel();
    }

    private async Task ReloadModel()
    {
        SetBusyState(true);
        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;
            _engine?.Dispose();
            _engine = new ColorizationEngine();

            await Task.Run(() => _engine.LoadModel(_currentModelPath!, useGpu));

            Log($"DDColor Loaded ({_engine.DeviceMode})");
            if (useGpu && _engine.DeviceMode.Contains("CPU")) ChkUseGpu.IsChecked = false;
            UpdateUi();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error: {ex.Message}");
            Log("Load Failed");
        }
        finally
        {
            SetBusyState(false);
        }
    }

    private void BtnOpenImage_Click(object sender, RoutedEventArgs e) => OpenImageDialog();
    private void BtnSave_Click(object sender, RoutedEventArgs e) => SaveOutputImage("colorized.png");

    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        if (_engine == null || _inputBitmap == null) return;

        SetBusyState(true);
        try
        {
            byte[] inputBytes = BitmapToBytes(_inputBitmap);
            byte[] resultBytes = await Task.Run(() => _engine.Process(inputBytes));
            ImgOutput.Source = BytesToBitmap(resultBytes);
            Log("Done.");
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error: {ex.Message}");
            Log("Error.");
        }
        finally
        {
            SetBusyState(false);
            UpdateUi();
        }
    }

    private void UpdateUi()
    {
        BtnOpenImage.IsEnabled = _engine != null;
        BtnRun.IsEnabled = _engine != null && _inputBitmap != null;
        BtnSave.IsEnabled = ImgOutput.Source != null;
    }
}
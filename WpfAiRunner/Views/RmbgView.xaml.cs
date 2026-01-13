using Microsoft.Win32;
using OnnxEngines.Rmbg;
using OnnxEngines.Utils;
using SixLabors.ImageSharp.PixelFormats;
using System.Windows;
using System.Windows.Controls;

namespace WpfAiRunner.Views;

public partial class RmbgView : BaseAiView
{
    private readonly RmbgEngine _engine = new();
    private string? _currentModelPath;

    protected override Image ControlImgInput => ImgInput;
    protected override Image? ControlImgOutput => ImgOutput;
    protected override ProgressBar? ControlPbarLoading => PbarLoading;
    protected override TextBlock? ControlTxtStatus => TxtStatus;

    public RmbgView() => InitializeComponent();
    public override void Dispose() => _engine.Dispose();

    protected override async void OnLoaded(RoutedEventArgs e)
    {
#if DEBUG
        if (string.IsNullOrEmpty(_currentModelPath))
        {
            string? debugPath = OnnxHelper.FindModelInDebug("rmbg-1.4.onnx");
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
        var dlg = new OpenFileDialog { Filter = "ONNX Model|*.onnx", Title = "Select RMBG-1.4 Model" };
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

    private void BtnOpenImage_Click(object sender, RoutedEventArgs e) => OpenImageDialog();
    private void BtnSave_Click(object sender, RoutedEventArgs e) => SaveOutputImage("rmbg_result.png");

    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        if (_inputBitmap == null) return;

        SetBusyState(true);
        try
        {
            float threshold = (float)SldThreshold.Value;
            Rgba32? bgColor = null;
            if (CboBackground.SelectedIndex > 0)
            {
                bgColor = CboBackground.SelectedIndex switch
                {
                    1 => new Rgba32(255, 255, 255),
                    2 => new Rgba32(0, 0, 0),
                    3 => new Rgba32(0, 255, 0),
                    4 => new Rgba32(0, 0, 255),
                    _ => null
                };
            }

            byte[] inputBytes = BitmapToBytes(_inputBitmap);
            byte[] resultBytes = await Task.Run(() => _engine.RemoveBackground(inputBytes, threshold, bgColor));

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
            UpdateUi();
        }
    }

    private void UpdateUi()
    {
        BtnLoadModel.IsEnabled = ControlPbarLoading?.Visibility != Visibility.Visible;
        BtnOpenImage.IsEnabled = !string.IsNullOrEmpty(_currentModelPath);
        BtnRun.IsEnabled = _inputBitmap != null && !string.IsNullOrEmpty(_currentModelPath);
        BtnSave.IsEnabled = ImgOutput.Source != null;
    }
}
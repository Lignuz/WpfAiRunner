using System.IO;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Win32;
using OnnxEngines.Depth;
using OnnxEngines.Utils;

namespace WpfAiRunner.Views;

public partial class DepthView : BaseAiView
{
    private DepthEstimator? _estimator;
    private string? _modelPath;
    private bool _hasInferenceResult = false;

    protected override Image ControlImgInput => ImgInput;
    protected override Image? ControlImgOutput => ImgOutput;
    protected override ProgressBar? ControlPbarLoading => PbarLoading;
    protected override TextBlock? ControlTxtStatus => TxtStatus;

    public DepthView() => InitializeComponent();
    public override void Dispose() => _estimator?.Dispose();

    protected override async void OnLoaded(RoutedEventArgs e)
    {
#if DEBUG
        if (_estimator == null && string.IsNullOrEmpty(_modelPath))
        {
            string? debugPath = OnnxHelper.FindModelInDebug("depth_anything_v2_small.onnx");
            if (debugPath != null)
            {
                await ReloadModelAsync(debugPath);
            }
        }
#endif
        UpdateButtons();
    }

    protected override void OnImageLoaded()
    {
        ImgOutput.Source = null;
        _hasInferenceResult = false;
        UpdateButtons();
    }

    private async void BtnPickModel_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "ONNX (*.onnx)|*.onnx" };
        if (dlg.ShowDialog(Window.GetWindow(this)) != true) return;
        await ReloadModelAsync(dlg.FileName);
    }

    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e)
    {
        if (!string.IsNullOrEmpty(_modelPath))
            await ReloadModelAsync(_modelPath);
    }

    private async Task ReloadModelAsync(string path)
    {
        SetBusyState(true);
        Log("Loading model...");
        await Task.Delay(10);

        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;
            _estimator?.Dispose();
            _estimator = null;

            _estimator = await Task.Run(() => new DepthEstimator(path, useGpu));
            _modelPath = path;
            _hasInferenceResult = false;
            ImgOutput.Source = null;

            TxtModel.Text = Path.GetFileName(path);
            Log($"Loaded on {_estimator.DeviceMode}");

            if (useGpu && _estimator.DeviceMode.Contains("CPU"))
            {
                ChkUseGpu.IsChecked = false;
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Load failed: {ex.Message}");
            Log("Load failed.");
        }
        finally
        {
            SetBusyState(false);
            UpdateButtons();
            GC.Collect();
        }
    }

    private void BtnOpenImage_Click(object sender, RoutedEventArgs e) => OpenImageDialog();

    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        if (_estimator == null || _inputBitmap == null) return;

        SetBusyState(true);
        Log("Estimating...");

        try
        {
            byte[] inputBytes = BitmapToBytes(_inputBitmap);
            await Task.Run(() => _estimator.RunInference(inputBytes));
            _hasInferenceResult = true;
            await UpdateResultImage();
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
            UpdateButtons();
            GC.Collect();
        }
    }

    private async void CboStyle_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_hasInferenceResult) return;
        SetBusyState(true);
        await UpdateResultImage();
        SetBusyState(false);
    }

    private async Task UpdateResultImage()
    {
        if (!_hasInferenceResult || _estimator == null) return;

        try
        {
            var style = (ColormapStyle)CboStyle.SelectedIndex;
            byte[] resultBytes = await Task.Run(() => _estimator.GetDepthMap(style));
            ImgOutput.Source = BytesToBitmap(resultBytes);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Style update failed: {ex.Message}");
        }
    }

    private void UpdateButtons()
    {
        bool busy = ControlPbarLoading?.Visibility == Visibility.Visible;
        BtnPickModel.IsEnabled = !busy;
        BtnOpenImage.IsEnabled = !busy;
        BtnRun.IsEnabled = !busy && _estimator != null && _inputBitmap != null;
        ChkUseGpu.IsEnabled = !busy;
    }
}
using System.Windows;
using System.Windows.Controls;
using WpfAiRunner.Views;

namespace WpfAiRunner;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        // 초기 화면
        MainContent.Content = new WelcomeView();
    }

    private void Menu_Click(object sender, RoutedEventArgs e)
    {
        if (sender is RadioButton rb && rb.Tag is string tag)
        {
            // 1. 기존 뷰 정리 (Dispose 호출)
            if (MainContent.Content is IDisposable oldView)
            {
                oldView.Dispose();
            }

            // 2. 새 뷰로 교체
            MainContent.Content = tag switch
            {
                "Home" => new Views.WelcomeView(),
                "Lama" => new Views.LamaView(),
                "Depth" => new Views.DepthView(),
                "Sam" => new Views.SamView(),   // MobileSAM / SAM 2 통합
                "Esrgan" => new Views.RealEsrganView(),
                "Rmbg" => new Views.RmbgView(),
                "Face" => new Views.FaceView(),
                "Anime" => new Views.AnimeGanView(),
                "Color" => new Views.ColorizationView(),
                "StableDiffusion" => new Views.SdView(),
                "OCR" => new Views.OcrView(),
                _ => new Views.WelcomeView()
            };
        }
    }
}
using System.Windows;

namespace LamaWpf;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        // Crash visibility in sample apps
        DispatcherUnhandledException += (_, ex) =>
        {
            MessageBox.Show(
                ex.Exception.ToString(),
                "Unhandled Exception",
                MessageBoxButton.OK,
                MessageBoxImage.Error);

            ex.Handled = true;
        };
    }
}

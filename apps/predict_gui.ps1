param(
    [string]$Python = 'python',
    [switch]$SmokeTest,
    [switch]$LegacyDemo
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

[xml]$layout = @'
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="DSA 三孔图案预测" Width="1160" Height="790" MinWidth="890" MinHeight="690"
        WindowStartupLocation="CenterScreen" Background="#F2F5FA" FontFamily="Microsoft YaHei UI">
  <Window.Resources>
    <Style TargetType="Button">
      <Setter Property="Padding" Value="12,5"/>
      <Setter Property="MinHeight" Value="30"/>
      <Setter Property="Margin" Value="0,0,8,0"/>
    </Style>
    <Style TargetType="TextBox">
      <Setter Property="MinHeight" Value="30"/>
      <Setter Property="VerticalContentAlignment" Value="Center"/>
      <Setter Property="Padding" Value="6,0"/>
    </Style>
    <Style x:Key="Preview" TargetType="Border">
      <Setter Property="Background" Value="White"/>
      <Setter Property="BorderBrush" Value="#D7DEEA"/>
      <Setter Property="BorderThickness" Value="1"/>
      <Setter Property="Padding" Value="12"/>
      <Setter Property="CornerRadius" Value="6"/>
    </Style>
  </Window.Resources>
  <Grid Margin="24">
    <Grid.RowDefinitions>
      <RowDefinition Height="Auto"/><RowDefinition Height="Auto"/><RowDefinition Height="Auto"/>
      <RowDefinition Height="*"/><RowDefinition Height="Auto"/>
    </Grid.RowDefinitions>
    <TextBlock Text="DSA 三孔图案预测" FontSize="27" FontWeight="Bold" Foreground="#142847" Margin="0,0,0,14"/>
    <Border x:Name="DropBox" Grid.Row="1" AllowDrop="True" Background="#E8F0FC" BorderBrush="#7F9BCA"
            BorderThickness="1" CornerRadius="6" Padding="14" Margin="0,0,0,12">
      <DockPanel>
        <Button x:Name="ChooseImage" DockPanel.Dock="Right" Content="选择图像…"/>
        <Button x:Name="Demo" DockPanel.Dock="Right" Content="填入演示样例"/>
        <StackPanel>
          <TextBlock Text="拖入一张已配准的 DSA 前模板图" FontWeight="SemiBold" FontSize="15" Foreground="#173967"/>
          <TextBlock x:Name="ImagePath" Text="尚未选择图像" Foreground="#425771" TextTrimming="CharacterEllipsis" Margin="0,4,10,0"/>
        </StackPanel>
      </DockPanel>
    </Border>
    <Grid Grid.Row="2" Margin="0,0,0,12">
      <Grid.RowDefinitions><RowDefinition Height="Auto"/><RowDefinition Height="Auto"/></Grid.RowDefinitions>
      <Grid.ColumnDefinitions><ColumnDefinition Width="Auto"/><ColumnDefinition Width="*"/><ColumnDefinition Width="Auto"/></Grid.ColumnDefinitions>
      <TextBlock Text="模型权重" VerticalAlignment="Center" Margin="0,0,12,0"/>
      <TextBox x:Name="Checkpoint" Grid.Column="1" Margin="0,0,10,0"/>
      <Button x:Name="ChooseCheckpoint" Grid.Column="2" Content="选择 .pt…"/>
      <WrapPanel Grid.Row="1" Grid.ColumnSpan="3" Margin="0,10,0,0">
        <TextBlock Text="像素标定（nm/px）" VerticalAlignment="Center" Margin="0,0,8,0"/>
        <TextBox x:Name="Spacing" Width="75" Margin="0,0,16,0"/>
        <TextBlock Text="输入类型" VerticalAlignment="Center" Margin="0,0,8,0"/>
        <ComboBox x:Name="Polarity" Width="136" MinHeight="30" SelectedIndex="0" Margin="0,0,16,0">
          <ComboBoxItem Content="自动" Tag="auto"/><ComboBoxItem Content="暗腔体" Tag="dark"/><ComboBoxItem Content="亮腔体" Tag="bright"/>
          <ComboBoxItem Content="二值模板（白腔体）" Tag="binary"/>
        </ComboBox>
        <TextBlock Text="ROI（可留空）" VerticalAlignment="Center" Margin="0,0,8,0"/>
        <TextBox x:Name="Roi" Width="155" ToolTip="左 上 宽 高，从 0 起算，以空格分隔" Margin="0,0,16,0"/>
        <Button x:Name="Run" Content="开始预测" Background="#214D8D" Foreground="White" FontWeight="Bold"/>
        <Button x:Name="OpenOutput" Content="打开输出目录" IsEnabled="False"/>
      </WrapPanel>
    </Grid>
    <Grid Grid.Row="3">
      <Grid.ColumnDefinitions><ColumnDefinition/><ColumnDefinition Width="12"/><ColumnDefinition/><ColumnDefinition Width="12"/><ColumnDefinition/></Grid.ColumnDefinitions>
      <Border Style="{StaticResource Preview}"><DockPanel>
        <TextBlock DockPanel.Dock="Top" Text="输入原图" FontWeight="Bold" Margin="0,0,0,10"/>
        <Image x:Name="InputPreview" Stretch="Uniform"/>
      </DockPanel></Border>
      <Border Grid.Column="2" Style="{StaticResource Preview}"><DockPanel>
        <TextBlock DockPanel.Dock="Top" Text="二值模板" FontWeight="Bold" Margin="0,0,0,10"/>
        <Image x:Name="TemplatePreview" Stretch="Uniform" RenderOptions.BitmapScalingMode="NearestNeighbor"/>
      </DockPanel></Border>
      <Border Grid.Column="4" Style="{StaticResource Preview}"><DockPanel>
        <TextBlock DockPanel.Dock="Top" Text="预测区域与原图叠加（黄色）" FontWeight="Bold" Margin="0,0,0,10"/>
        <Image x:Name="OverlayPreview" Stretch="Uniform"/>
      </DockPanel></Border>
    </Grid>
    <StackPanel Grid.Row="4" Margin="0,12,0,0">
      <TextBlock x:Name="Classification" FontSize="19" FontWeight="Bold" Foreground="#142847" Text="分类结果待预测"/>
      <TextBlock x:Name="Scores" Foreground="#425771" Margin="0,4,0,0" TextWrapping="Wrap"/>
      <TextBlock x:Name="Status" Foreground="#425771" Margin="0,4,0,0" TextWrapping="Wrap"
                 Text="请使用原图标定。预测后可从输出目录查看分割边界核对图。"/>
    </StackPanel>
  </Grid>
</Window>
'@
$reader = [System.Xml.XmlNodeReader]::new($layout)
$window = [Windows.Markup.XamlReader]::Load($reader)
$ui = @{}
foreach ($name in 'DropBox','ChooseImage','Demo','ImagePath','Checkpoint','ChooseCheckpoint','Spacing','Polarity','Roi',
    'Run','OpenOutput','InputPreview','TemplatePreview','OverlayPreview','Classification','Scores','Status') {
    $ui[$name] = $window.FindName($name)
}

$demoImage = Join-Path $projectRoot 'data/2D_RE_R25_RH32_统一200nm_256/3孔/F004_R01_C01.png'
$demoCheckpoint = Join-Path $projectRoot 'experiments/reconstruction/sem_real_256_balanced/best.pt'
$semDemo = -not $LegacyDemo -and (Test-Path -LiteralPath $demoImage) -and (Test-Path -LiteralPath $demoCheckpoint)
if (-not $semDemo) {
    $demoImage = Join-Path $projectRoot 'data/sem_r2_c07/sem_r2_c07_local_pore_removal_input_128.png'
    $demoCheckpoint = Join-Path $projectRoot 'experiments/archive/demo-train-v2/best.pt'
}
$deployment = Join-Path $projectRoot 'experiments/deployment/best.pt'
if ($semDemo -and (Test-Path -LiteralPath $deployment)) { $demoCheckpoint = $deployment }
$ui.Demo.IsEnabled = (Test-Path -LiteralPath $demoImage) -and (Test-Path -LiteralPath $demoCheckpoint)
$labels = [ordered]@{
    good='合格三孔'; line_or_bridge='线形或桥连'; wrong_count='孔数异常'; other_defect='其他缺陷'
}
$script:selectedImage = $null
$script:output = $null
$script:activeJob = $null
$script:smokePassed = $false

function Select-Image([string]$path) {
    $script:selectedImage = (Get-Item -LiteralPath $path).FullName
    $ui.ImagePath.Text = $script:selectedImage
    $ui.ImagePath.ToolTip = $script:selectedImage
    $ui.TemplatePreview.Source = $null
    $ui.OverlayPreview.Source = $null
    $ui.Classification.Text = '分类结果待预测'
    $ui.Scores.Text = ''
    $ui.OpenOutput.IsEnabled = $false
    try {
        $ui.InputPreview.Source = Read-Preview $script:selectedImage
    } catch {
        $ui.InputPreview.Source = $null
    }
}
function Pick-File([string]$filter) {
    $dialog = [Microsoft.Win32.OpenFileDialog]::new()
    $dialog.Filter = $filter
    if ($dialog.ShowDialog($window)) { return $dialog.FileName }
}
function Set-Example {
    Select-Image $demoImage
    $ui.Checkpoint.Text = $demoCheckpoint
    $ui.Roi.Clear()
    if ($semDemo) {
        $ui.Spacing.Text = '0.78125'
        $ui.Polarity.SelectedIndex = 0
        $ui.Status.Text = '已填入留出集的灰度 SEM 图和 256 × 256 模型，可演示原图二值化流程。'
    } else {
        $ui.Spacing.Text = '1'
        $ui.Polarity.SelectedIndex = 1
        $ui.Status.Text = '演示图和合成权重已填入。1 nm/px 为演示设定。'
    }
}
function Set-Busy([bool]$busy) {
    foreach ($name in 'ChooseImage','Checkpoint','ChooseCheckpoint','Spacing','Polarity','Roi','Run') {
        $ui[$name].IsEnabled = -not $busy
    }
    $ui.DropBox.AllowDrop = -not $busy
    $ui.Demo.IsEnabled = (-not $busy) -and (Test-Path -LiteralPath $demoImage) -and (Test-Path -LiteralPath $demoCheckpoint)
}
function Read-Preview([string]$path) {
    $bitmap = [System.Windows.Media.Imaging.BitmapImage]::new()
    $bitmap.BeginInit()
    $bitmap.CacheOption = [System.Windows.Media.Imaging.BitmapCacheOption]::OnLoad
    $bitmap.UriSource = [Uri]::new($path)
    $bitmap.EndInit()
    $bitmap.Freeze()
    return $bitmap
}
function Show-Result([string]$folder) {
    $result = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $folder 'prediction/prediction.json') | ConvertFrom-Json
    $ui.InputPreview.Source = Read-Preview (Join-Path $folder 'input_preview.png')
    $ui.TemplatePreview.Source = Read-Preview (Join-Path $folder 'segmentation/template.png')
    $ui.OverlayPreview.Source = Read-Preview (Join-Path $folder 'overlay_on_input.png')
    $scoreTitle = '分类分数（未校准）  '
    if ($result.model_kind -eq 'sem_grayscale_256_v1') {
        $ui.Classification.Text = "预测孔数（分类）  $($result.prediction)"
        $shownLabels = [ordered]@{'2孔'='两孔'; '3孔'='三孔'}
    } else {
        $ui.Classification.Text = "分类  $($labels[$result.prediction])    连通开口数  $($result.component_count)"
        $shownLabels = $labels
    }
    $ui.Scores.Text = $scoreTitle + (($shownLabels.Keys | ForEach-Object {
        $value = $result.class_scores_uncalibrated.PSObject.Properties[$_].Value
        "$($shownLabels[$_]) $($value.ToString('P1'))"
    }) -join '   ·   ')
    $ui.Status.Text = '预测完成。可打开输出目录查看详细结果。'
    $script:output = $folder
    $ui.OpenOutput.IsEnabled = $true
}

$timer = [System.Windows.Threading.DispatcherTimer]::new()
$timer.Interval = [TimeSpan]::FromMilliseconds(200)
$timer.Add_Tick({
    if (-not $script:activeJob.Process.HasExited) { return }
    $timer.Stop()
    $job = $script:activeJob
    $script:activeJob = $null
    try {
        $stderr = $job.Stderr.GetAwaiter().GetResult()
        if ($job.Process.ExitCode -ne 0) {
            $detail = $stderr.Trim()
            if (-not $detail) { $detail = $job.Stdout.GetAwaiter().GetResult().Trim() }
            throw "预测程序退出码 $($job.Process.ExitCode)  $detail"
        }
        Show-Result $job.Output
        $script:smokePassed = $true
    } catch {
        $ui.Classification.Text = '预测未完成'
        $ui.Status.Text = $_.Exception.Message
        if ($SmokeTest) { [Console]::Error.WriteLine($_.Exception.Message) }
    } finally {
        $job.Process.Dispose()
        Set-Busy $false
        if ($SmokeTest) { $window.Close() }
    }
})

function Start-Prediction {
    try {
        if (-not $script:selectedImage) { throw '请先拖入或选择一张模板图。' }
        $weightPath = $ui.Checkpoint.Text.Trim()
        if (-not (Test-Path -LiteralPath $weightPath -PathType Leaf)) { throw '请选择有效的模型权重文件。' }
        $culture = [Globalization.CultureInfo]::InvariantCulture
        $spacing = 0.0
        if (-not [double]::TryParse($ui.Spacing.Text,[Globalization.NumberStyles]::Float,$culture,[ref]$spacing) -or
            $spacing -le 0 -or [double]::IsNaN($spacing) -or [double]::IsInfinity($spacing)) {
            throw '请输入有效的像素标定，例如 0.8。'
        }
        $folder = Join-Path $projectRoot ('outputs/gui/gui-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,6))
        $argsList = @('-m','dsa','predict-image','--image',$script:selectedImage,
            '--checkpoint',(Get-Item -LiteralPath $weightPath).FullName,
            '--nm-per-px',$spacing.ToString($culture),'--out',$folder,'--dark-only-overlay')
        $inputType = [string]$ui.Polarity.SelectedItem.Tag
        if ($inputType -eq 'binary') {
            $argsList += '--binary-template'
        } else {
            $argsList += @('--polarity',$inputType)
        }
        if ($ui.Roi.Text.Trim()) {
            $coords = $ui.Roi.Text.Trim() -split '[\s,，]+'
            if ($coords.Count -ne 4 -or ($coords | Where-Object { $_ -notmatch '^\d+$' })) {
                throw 'ROI 需要四个整数，按左、上、宽、高输入，例如 320 140 128 160。'
            }
            $argsList += @('--roi') + $coords
        }
        $info = [Diagnostics.ProcessStartInfo]::new()
        $info.FileName = $Python
        $info.WorkingDirectory = $projectRoot
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $info.StandardOutputEncoding = [Text.Encoding]::UTF8
        $info.StandardErrorEncoding = [Text.Encoding]::UTF8
        $info.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8'
        if ($info.PSObject.Properties['ArgumentList']) {
            foreach ($arg in $argsList) { $info.ArgumentList.Add([string]$arg) }
        } else {
            # These arguments are options, numeric values, or existing Windows file paths.
            $info.Arguments = ($argsList | ForEach-Object { '"' + $_ + '"' }) -join ' '
        }
        $process = [Diagnostics.Process]::Start($info)
        $script:activeJob = @{
            Process = $process; Output = $folder
            Stdout = $process.StandardOutput.ReadToEndAsync(); Stderr = $process.StandardError.ReadToEndAsync()
        }
        foreach ($name in 'TemplatePreview','OverlayPreview') { $ui[$name].Source = $null }
        $ui.OpenOutput.IsEnabled = $false
        $ui.Classification.Text = '正在分割和预测…'
        $ui.Scores.Text = ''
        $ui.Status.Text = '模型正在后台运行。'
        Set-Busy $true
        $timer.Start()
    } catch {
        $ui.Classification.Text = '预测未完成'
        $ui.Status.Text = $_.Exception.Message
        if ($SmokeTest) { [Console]::Error.WriteLine($_.Exception.Message); $window.Close() }
    }
}

$ui.DropBox.Add_PreviewDragOver({
    param($sender,$e)
    $e.Effects = if ($e.Data.GetDataPresent([Windows.DataFormats]::FileDrop)) { [Windows.DragDropEffects]::Copy } else { [Windows.DragDropEffects]::None }
    $e.Handled = $true
})
$ui.DropBox.Add_PreviewDrop({
    param($sender,$e)
    $paths = @($e.Data.GetData([Windows.DataFormats]::FileDrop))
    if ($paths.Count -eq 1 -and $paths[0] -and (Test-Path -LiteralPath $paths[0] -PathType Leaf)) {
        Select-Image $paths[0]
        $ui.Status.Text = '图像已载入。确认权重和标定后开始预测。'
    } else { $ui.Status.Text = '请一次拖入一个图像文件。' }
    $e.Handled = $true
})
$ui.ChooseImage.Add_Click({
    $path = Pick-File '图像|*.png;*.tif;*.tiff;*.jpg;*.jpeg;*.bmp|所有文件|*.*'
    if ($path) { Select-Image $path }
})
$ui.ChooseCheckpoint.Add_Click({
    $path = Pick-File '模型检查点|*.pt;*.pth|所有文件|*.*'
    if ($path) { $ui.Checkpoint.Text = $path }
})
$ui.Demo.Add_Click({ Set-Example })
$ui.Run.Add_Click({ Start-Prediction })
$ui.OpenOutput.Add_Click({ if ($script:output) { Invoke-Item -LiteralPath $script:output } })
$window.Add_Closing({
    param($sender,$e)
    if ($script:activeJob) {
        $ui.Status.Text = '预测仍在运行，完成后即可关闭窗口。'
        $e.Cancel = $true
    }
})
if ($SmokeTest) {
    $window.Add_Loaded({ Set-Example; Start-Prediction })
}
[void]$window.ShowDialog()
if ($SmokeTest) {
    if (-not $script:smokePassed) { exit 1 }
    Write-Output $script:output
}

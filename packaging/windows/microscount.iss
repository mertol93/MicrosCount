; Inno Setup script for MicrosCount (built by packaging\build_windows.ps1)
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#define MyAppName "MicrosCount"
#define MyAppExe "MicrosCount.exe"
#define MyAppURL "https://github.com/mertol93/microscount"

[Setup]
AppId={{6F2A9C1E-4B7D-4E3A-9D5F-8C1B2E7A4D90}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher=Mertol Tüfekci
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; installs for the current user without administrator rights; users may choose "all users"
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
LicenseFile=..\..\LICENSE
OutputDir=..\..\dist\release
OutputBaseFilename=MicrosCount-{#MyAppVersion}-Windows-x64-Setup
SetupIconFile=..\icons\microscount.ico
UninstallDisplayIcon={app}\{#MyAppExe}
UninstallDisplayName={#MyAppName} {#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\..\dist\MicrosCount\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"
Name: "{autoprograms}\{#MyAppName} (command line)"; Filename: "{cmd}"; Parameters: "/k ""{app}\microscount-cli.exe"" --help"; WorkingDir: "{userdocs}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

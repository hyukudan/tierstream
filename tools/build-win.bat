@echo off
rem Windows build: tools\build-win.bat [cpu|cuda]  ->  build\ or build-cuda\
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
cd /d %~dp0..
if /i "%1"=="cuda" (
  cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DSS_CUDA=ON || exit /b 1
  cmake --build build-cuda || exit /b 1
) else (
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release || exit /b 1
  cmake --build build || exit /b 1
)

@echo off
if not defined VULKAN_SDK (
    echo Error: VULKAN_SDK environment variable is not set
    pause
    exit /b 1
)

echo Compiling vertex shader...
"%VULKAN_SDK%\Bin\glslangValidator.exe" -V shader.vert -o vert.spv
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to compile vertex shader
    pause
    exit /b 1
)

echo Compiling fragment shader...
"%VULKAN_SDK%\Bin\glslangValidator.exe" -V shader.frag -o frag.spv
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to compile fragment shader
    pause
    exit /b 1
)

echo Shaders compiled successfully
pause 
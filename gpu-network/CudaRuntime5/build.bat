@echo off
nvcc kernel.cu -o kernel.exe ^
-ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64" ^
-I"C:\Users\kcjan\vsincludes\raylib-5.5_win64_msvc16\include" ^
-L"C:\Users\kcjan\vsincludes\raylib-5.5_win64_msvc16\lib" ^
-lraylib -lopengl32 -lgdi32 -lwinmm -lkernel32 -luser32 -lshell32 -lcomdlg32 -lole32 -loleaut32 -luuid ^
-llegacy_stdio_definitions -lvcruntime -lucrt -lmsvcrt ^
--use_fast_math -Xcompiler "/openmp" -O3

kernel.exe

First walking trials on CUDA programming.

# Set up
## Windows 10
1. install VS Express 11
2. install CUDA toolkit 9.2
3. add `C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin` to `PATH`
4. copy contents from `C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\x86_amd64` to `C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\amd64` (create `amd64` folder)
5. Navigate to `amd64` folder and copy `vcvarsx86_amd64.bat` to `vcvars64.bat`

After those steps you can use `nvcc`:
```
nvcc hello_world.cu -o bin/hello_world
```
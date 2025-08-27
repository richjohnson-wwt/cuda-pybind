## CUDA-Pybind 

This app demonstrates how to do two vector operations that are done in C++ on GPU but setup and called from Python on CPU. 

The main benefits of programming CUDA in C++ are 
1. performance
2. low-level control
3. direct access to GPU hardware

The benefits of using Python with TensorFlow are 
1. development speed
2. ease of use
3. rich ecosystem

#### Intial Setup - Do every time a new VM is started

    sudo apt install cmake  # if not already installed
    sudo apt install nvidia-cuda-toolkit  # if not already installed
    uv venv
    source .venv/bin/activate
    uv pip install conan
    uv pip install numpy pytest
    conan profile detect
    vi ~/.gitconfig

    [user]
        email = rich.johnson@wwt.com
        name = Rich Johnson


#### Debug Config

    conan install . \
        --output-folder=build/debug \
        --build=missing \
        --settings=build_type=Debug

    cd build/debug
    cmake ../.. -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug    
    cmake --build .

    # cmake --preset conan-debug
    # cmake --build build/debug

To run via Python:

    PYTHONPATH=build/debug python src/app.py

To run via C++:

    ./build/debug/cuda-pybind

#### Release Config

    conan install . \
        --output-folder=build/release \
        --build=missing \
        --settings=build_type=Release

    cmake --preset conan-release
    cmake --build --preset conan-release

    PYTHONPATH=build/release python src/app.py


#### Test via Python pytest (from root)

    PYTHONPATH=build/debug pytest src/test_vector_ops.py

#### Test via C++ Catch2 (inside build/debug folder)

    ./test/test_vector_add 

    or

    ctest
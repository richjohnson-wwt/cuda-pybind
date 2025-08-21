## CUDA-Pybind 

This app demonstrates how to integrate Python and C++ in a CUDA application where Python prepares the data and C++ executes the computations on GPU

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

    cmake --preset conan-debug
    cmake --build build/debug

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
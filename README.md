## CUDA-Pybind 

This app demonstrates how to integrate Python and C++ in a CUDA application where Python prepares the data and C++ executes the computations on GPU

#### Intial Setup - Do every time a new VM is started

    uv venv
    source .venv/bin/activate
    uv pip install conan
    uv pip install numpy
    conan profile detect
    vi ~/.gitconfig

    [user]
        email = rich.johnson@wwt.com
        name = Rich Johnson


#### Debug Config

    conan install . --output-folder=build/debug --build=missing --settings=build_type=Debug

    cmake --preset conan-debug
    cmake --build build/debug

    Old way:
    cd build/debug
    cmake ../.. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug 
    cmake --build .

    PYTHONPATH=build/debug python src/app.py

#### Release Config

    conan install . \
        --output-folder=build/release \
        --build=missing \
        --settings=build_type=Release \
        --generator CMakeDeps \
        --generator CMakeToolchain

    cmake --preset conan-release
    cmake --build --preset conan-release

    PYTHONPATH=build/release python src/app.py

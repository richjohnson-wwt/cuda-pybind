# Intial Setup - Do every time a new VM is started

    uv venv
    source .venv/bin/activate
    uv pip install conan


# Debug Config

    conan install . --output-folder=build/debug --build=missing --settings=build_type=Debug

    cd build/debug
    cmake ../.. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug 
    cmake --build .

    PYTHONPATH=build/debug python src/app.py
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------
# 1. Install Dependencies
# ---------------------------
RUN apt-get update && apt-get install -y \
    locales build-essential cmake git pkg-config \
    libusb-1.0-0-dev libudev-dev libglfw3-dev \
    libgl1-mesa-dev libglu1-mesa-dev libx11-dev \
    python3 python3-pip python3-dev python3-pybind11 sudo \
    libxcb-xinerama0 libxkbcommon-x11-0 libxcb-cursor0 libqt5x11extras5 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# 2. Setup Locale
# ---------------------------
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

WORKDIR /intern_MuroranIT

# ---------------------------
# 3. ONLY Copy & Build the SDK (This creates the permanent cache!)
# ---------------------------
# Docker will cache this step forever unless you edit files inside python/pyorbbecsdk
COPY python/pyorbbecsdk python/pyorbbecsdk

RUN if [ -d "python/pyorbbecsdk" ]; then \
        cd python/pyorbbecsdk && \
        pip3 install --upgrade pip && \
        pip3 install -r requirements.txt wheel && \
        mkdir -p build && cd build && \
        cmake -Dpybind11_DIR=$(pybind11-config --cmakedir) .. && \
        make -j$(nproc) && \
        make install && \
        cd .. && \
        rm -rf dist build && \
        python3 setup.py bdist_wheel && \
        pip3 install dist/*.whl; \
    else \
        echo "Error: python/pyorbbecsdk directory not found!" && exit 1; \
    fi
# ---------------------------
# 3.5 Install App Dependencies (Super Fast!)
# ---------------------------
# Put all your new python libraries here! Because this is a separate step 
# AFTER Step 3, Docker will use the cached C++ build and only take 
# 2 seconds to install your new libraries.
RUN pip3 install Pillow imutils scipy

# ---------------------------
# 4. Copy the Rest of Your Workspace
# ---------------------------
# Now copy your python scripts. If you edit a script, Docker only re-runs 
# from this line downward, taking less than a second!
COPY . .

# ---------------------------
# 5. Environment
# ---------------------------
ENV LD_LIBRARY_PATH=/intern_MuroranIT/python/pyorbbecsdk/install/lib

CMD ["bash"]
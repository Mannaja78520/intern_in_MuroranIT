Building From Source    
Ubuntu LTS

    Ensure apt-get is up to date

    sudo apt-get update && sudo apt-get upgrade Note: Use sudo apt-get dist-upgrade, instead of sudo apt-get upgrade, in case you have an older Ubuntu 14.04 version

    Install Python and its development files via apt-get

    sudo apt-get install python3 python3-dev

    Clone the librealsense repository and navigate into the directory:
        cd ~
        git clone https://github.com/realsenseai/librealsense.git
        cd librealsense
    Configure and make:

    mkdir build
    cd build
    cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=$(which python3)
    make -j4
    sudo make install

    Note: For building a self-contained (statically compiled) pyrealsense2 library add the CMake flag:

    -DBUILD_SHARED_LIBS=false

    Note: To force compilation with a specific version on a system with a few Python versions installed, add the following flag to CMake command:

    -DPYTHON_EXECUTABLE=[full path to the exact python executable]

    update your PYTHONPATH environment variable to add the path to the pyrealsense library

    export PYTHONPATH=$PYTHONPATH:/usr/local/lib

    Note: If this doesn't work, try using the following path instead: export PYTHONPATH=$PYTHONPATH:/usr/local/lib/[python version]/pyrealsense2

    Alternatively, copy the build output (librealsense2.so and pyrealsense2.so) next to your script. Note: Python 3 module filenames may contain additional information, e.g. pyrealsense2.cpython-35m-arm-linux-gnueabihf.so

After build it:
    Copy udev rule:

        cd ~/realsense/librealsense
        sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/

    Reload rule:

        sudo udevadm control --reload-rules
        sudo udevadm trigger

    Add user to groups:

        sudo usermod -aG plugdev $USER
        sudo usermod -aG video $USER

    Create rule to fix iio permission:

        sudo nano /etc/udev/rules.d/99-iio.rules
        Add:
            KERNEL=="iio*", MODE="0666"
        
    Reload rule:

        sudo udevadm control --reload-rules
        sudo udevadm trigger

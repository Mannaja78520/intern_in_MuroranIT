# How to install by Man

## get source code

# 
git clone https://github.com/orbbec/pyorbbecsdk.git 
#

## Install Dependencies

# 
sudo apt-get install python3-dev python3-venv python3-pip python3-opencv 
#

## Build the Project

# 
cd pyorbbecsdk
pip3 install -r requirements.txt
mkdir build
cd build
cmake -Dpybind11_DIR=`pybind11-config --cmakedir` ..
make -j4
make install 
#

## insall udev rule

#
cd pyorbbecsdk
export PYTHONPATH=$PYTHONPATH:$(pwd)/install/lib/
sudo bash ./scripts/install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
#

## Making a Python Wheel

#
cd pyorbbecsdk
python3 -m venv ./venv
source venv/bin/activate
pip3 install -r requirements.txt
mkdir build
cd build
cmake -Dpybind11_DIR=`pybind11-config --cmakedir` ..
make -j4
make install
cd ..
pip3 install wheel
python3 setup.py bdist_wheel
pip3 install dist/*.whl
#


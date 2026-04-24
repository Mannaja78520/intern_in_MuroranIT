# First of all if need to run this all file plese install roboflow and ultralytic
#
pip install -U roboflow ultralytics
#

# If need to train own model
#
    1.capture all of the pictures that you can see the thing which you need to train.
    2.label it in roboflow (only the thing that you need to train).
    3.get download code in the version tab.
    4.go to the directory which have the download file from download source code in roboflow
    5.run the download file from roboflow
    6.when you got all of the file run edit the train model file use the name of your project front of the data.yaml like:
        muroranit_t-hook_combined/data.yaml
    7.wait until it finished the trainnig.
    8.your model it in the floder runs/train/weights
    9.use that model in the another program.
#

# Run the model
#
    1.put the model inside directory which need to run the YOLO program.
    2.edit the model name in the YOLO program make it match same as the model which you need to use.
    3.run the YOLO program.
#
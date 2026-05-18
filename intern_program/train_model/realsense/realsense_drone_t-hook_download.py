from roboflow import Roboflow
rf = Roboflow(api_key="RV66efGRV1dBEXPMeWFX")
project = rf.workspace("i-a").project("t-hook_crazyfile-drone")
version = project.version(1)
dataset = version.download("yolo26")
                
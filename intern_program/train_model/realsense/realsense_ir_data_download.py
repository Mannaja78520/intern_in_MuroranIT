from roboflow import Roboflow
rf = Roboflow(api_key="vWHDwSEhkRVq7eVrjVV3")
project = rf.workspace("m-k-8ngn3").project("muroranit_t-hook_ir")
version = project.version(1)
dataset = version.download("yolo26")
                
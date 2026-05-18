# intern_in_MuroranIT

## If you need to use something in this floder it have README.md in the separate floder

## Group docker to use it
sudo usermod -aG docker cv
newgrp docker
groups

## Build docker
sudo docker build -t intern-muroran-it-app:latest .




## Run docker
xhost +local:docker
docker run -it --rm \
--privileged \
--net=host \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /dev:/dev \
-v /dev/bus/usb:/dev/bus/usb \
intern-muroran-it-app:latest

## Run docker with save video
xhost +local:docker
docker run -it --rm --privileged --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev:/dev \
  -v /dev/bus/usb:/dev/bus/usb \
  -v $(pwd)/recordings:/intern_MuroranIT/recordings \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  --user $(id -u):$(id -g) \
  intern-muroran-it-app:latest

## How to clear Docker
docker builder prune -f
docker system prune -a --volumes -f



## stop all container
docker stop $(docker ps -aq) 2>/dev/null

## delete all container
docker rm $(docker ps -aq) 2>/dev/null

## delete all docker images
docker rmi $(docker images -q) 2>/dev/null

## delete build cache
docker builder prune -a -f

## delete all (network + volume)
docker system prune -a --volumes -f

docker ps -a
docker images
docker volume ls

## Build (no cache)
docker build --no-cache -t intern-muroran-it-app:latest .



The remaining ~680 ms for add is the irreducible cost of creating one genuinely heavy card — if you want add fully instant too, the next step would be lazily building the Combine/OVERLAY/YOLO sub-sections only when that step Type is selected. I'll report the final stress-test numbers once the run finishes.

make it

in closing and opening did it need to config the erose and dialate? why when i try close it like noting?

and when we use ctrl+s to save or load the file the string say the path it too long and make the program widler make the main 6 picture shift but the pipeline setting are in the same place as before that make hard to config while see the main picture

the only lab method it can detect the red clearly than HSV but it not good enough how it be can you suggestion how to do or make it it can classify clearly in black blackground but in white background sometime it dissapear or when adj it it see the background too because the backgrond is not only 1 color sometime it white sometime back sometime the red are between the black and white.

and when rec the video save the config in the problem while rec to the folder and rec all of 6 picture in the same video too.

when use it have some warning this it problem??
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame2.!labelframe2.!labelframe5.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1243, in _refresh_visibility
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame2.!labelframe2.!labelframe3.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1243, in _refresh_visibility
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame2.!labelframe2.!labelframe.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1314, in _refresh_yolo_input_visibility
    img_pick_fr.pack(fill="x", before=yolo_fr_mode_row)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame2.!labelframe2.!labelframe12.!labelframe.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1314, in _refresh_yolo_input_visibility
    img_pick_fr.pack(fill="x", before=yolo_fr_mode_row)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame2.!labelframe2.!labelframe10.!labelframe.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1314, in _refresh_yolo_input_visibility
    img_pick_fr.pack(fill="x", before=yolo_fr_mode_row)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame2.!labelframe2.!labelframe8.!labelframe.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1314, in _refresh_yolo_input_visibility
    img_pick_fr.pack(fill="x", before=yolo_fr_mode_row)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame2.!labelframe2.!labelframe6.!labelframe.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1314, in _refresh_yolo_input_visibility
    img_pick_fr.pack(fill="x", before=yolo_fr_mode_row)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame2.!labelframe2.!labelframe4.!labelframe.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1314, in _refresh_yolo_input_visibility
    img_pick_fr.pack(fill="x", before=yolo_fr_mode_row)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame2.!labelframe2.!labelframe2.!labelframe.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1243, in _refresh_visibility
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame2.!labelframe2.!labelframe12.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1243, in _refresh_visibility
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame2.!labelframe2.!labelframe10.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1243, in _refresh_visibility
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame2.!labelframe2.!labelframe8.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1243, in _refresh_visibility
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame2.!labelframe2.!labelframe6.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1243, in _refresh_visibility
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame2.!labelframe2.!labelframe4.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1243, in _refresh_visibility
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame2.!labelframe2.!labelframe2.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1229, in _refresh_visibility
    combine_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame.!labelframe2.!labelframe14.!frame4"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1229, in _refresh_visibility
    combine_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame.!labelframe2.!labelframe12.!frame4"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1229, in _refresh_visibility
    combine_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame.!labelframe2.!labelframe10.!frame4"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1229, in _refresh_visibility
    combine_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame.!labelframe2.!labelframe8.!frame4"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1229, in _refresh_visibility
    combine_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame.!labelframe2.!labelframe6.!frame4"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1229, in _refresh_visibility
    combine_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame.!labelframe2.!labelframe4.!frame4"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1229, in _refresh_visibility
    combine_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame.!labelframe2.!labelframe2.!frame4"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1243, in _refresh_visibility
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame.!labelframe2.!labelframe14.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1243, in _refresh_visibility
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame.!labelframe2.!labelframe12.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1243, in _refresh_visibility
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame.!labelframe2.!labelframe10.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1243, in _refresh_visibility
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame.!labelframe2.!labelframe8.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1243, in _refresh_visibility
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame.!labelframe2.!labelframe6.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1243, in _refresh_visibility
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame.!labelframe2.!labelframe4.!frame3"
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1262, in _sync_kind
    _refresh_visibility()
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/pipeline_ui_mixin.py", line 1243, in _refresh_visibility
    morph_fr.pack(fill="x", before=yolo_fr)
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 2483, in pack_configure
    self.tk.call(
_tkinter.TclError: bad window path name ".!frame.!canvas.!frame.!labelframe.!notebook.!frame.!labelframe2.!labelframe2.!frame3"
^CException in Tkinter callback
Traceback (most recent call last):
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1968, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 862, in callit
    func(*args)
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/realsense_video_analyzer.py", line 3226, in _play_loop
    self._process(f_rgb, f_ir)
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/processing_mixin.py", line 260, in _process
    m1, m2, red_mask = self._rgb_detect(hsv, lab, h1l, h1h, h2l, h2h,
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/processing_mixin.py", line 176, in _rgb_detect
    labm = self._lab_mask(lab)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/processing_mixin.py", line 146, in _lab_mask
    return cv2.inRange(lab, (0, amn, bmn), (255, amx, bmx))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
^C^CTraceback (most recent call last):
  File "/home/mannaja/intern_in_MuroranIT/intern_program/realsense/2026_05_18/realsense_video_analyzer.py", line 3267, in <module>
    root.mainloop()
  File "/home/mannaja/miniconda3/envs/intern_muroranIT_py312/lib/python3.12/tkinter/__init__.py", line 1505, in mainloop
    self.tk.mainloop(n)
KeyboardInterrupt
(intern_muroranIT_py312) mannaja@mannaja:~/intern_in_MuroranIT/intern_program/realsense/2026_05_18$ 

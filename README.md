# intern_in_MuroranIT

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



# หยุด container ทั้งหมด
docker stop $(docker ps -aq) 2>/dev/null

# ลบ container ทั้งหมด
docker rm $(docker ps -aq) 2>/dev/null

# ลบ image ทั้งหมด
docker rmi $(docker images -q) 2>/dev/null

# ลบ build cache
docker builder prune -a -f

# ลบทุกอย่าง (network + volume)
docker system prune -a --volumes -f



docker ps -a
docker images
docker volume ls



# Build (no cache)
docker build --no-cache -t intern-muroran-it-app:latest .
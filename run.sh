# Enable GUI 
xhost +

# Imstall 
docker build -t repair_projv2 .



# Run 
docker run --privileged -it --gpus all --userns host \
  -p 55555:22 --net=host \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --device /dev/video0:/dev/video0 \
  --volume="$HOME/.Xauthority:/home/developer/.Xauthority:rw" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  repair_projv2
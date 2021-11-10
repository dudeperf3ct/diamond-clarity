export DISPLAY=:1
xhost +
sudo docker run --rm -it --gpus all \
    -v $(pwd):/app \
    --ipc=host \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -p 8888:8888 \
    --name dc_exp \
    dc bash
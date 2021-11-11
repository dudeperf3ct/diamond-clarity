export DISPLAY=:1
xhost +
sudo docker run --rm -it --gpus all \
    -v $(pwd):/app \
    --ipc=host \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -p 8878:8878 \
    --name dc_tf_exp \
    dc_tf bash
FROM nvcr.io/nvidia/tensorflow:21.10-tf2-py3
ENV DEBIAN_FRONTEND=noninteractive

RUN pip3 install --upgrade pip
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
# copy everything else
COPY . /app
CMD bash
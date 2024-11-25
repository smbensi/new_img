FROM python:3.8-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# RUN pip install "tritonclient[http]==2.41.0"
WORKDIR /code  

COPY gevent-24.2.1-cp38-cp38-linux_aarch64.whl /code/gevent-24.2.1-cp38-cp38-linux_aarch64.whl
COPY requirements.txt /code/requirements.txt

RUN pip install gevent-24.2.1-cp38-cp38-linux_aarch64.whl
RUN pip install -r requirements.txt
# COPY . .
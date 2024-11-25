FROM python:3.8.20


RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop libgl1 \
    libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0

RUN python3 -m pip install --upgrade pip wheel
RUN mkdir /code
WORKDIR /code

COPY gevent-24.2.1-cp38-cp38-linux_aarch64.whl /code/gevent-24.2.1-cp38-cp38-linux_aarch64.whl
RUN pip install gevent-24.2.1-cp38-cp38-linux_aarch64.whl


COPY requirements.txt .
RUN pip install  -r requirements.txt

COPY . .
RUN pip install -e .
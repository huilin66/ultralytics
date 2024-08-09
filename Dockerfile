# Ultralytics YOLO 🚀, AGPL-3.0 license
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV MKL_THREADING_LAYER=GNU

# Downloads to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

RUN apt update \
    && apt install --no-install-recommends -y gcc git zip unzip wget curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 libsm6

RUN apt upgrade --no-install-recommends -y openssl tar

WORKDIR /workspace

RUN python3 -m pip install --upgrade pip wheel

RUN pip install --no-cache-dir numpy==1.23.5
RUN pip install ultralytics
RUN pip install scikit-image
RUN pip install imagecodecs
# Remove exported models
RUN rm -rf tmp

COPY ultralytics ultralytics
COPY tz2yolo.py tz2yolo.py
COPY run.py run.py
COPY v3_0808_01.pt v3_0808_01.pt

CMD ["python", "run.py", "/input_path", "/output_path"]


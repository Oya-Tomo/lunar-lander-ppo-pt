FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

RUN python3 -m venv /venv

RUN /venv/bin/python3 -m pip install \
    torch torchvision \
    numpy \
    gymnasium \
    pygame \
    swig \
    box2d \
    wandb

ENV PATH="/venv/bin:$PATH"
ENV VIRTUAL_ENV="/venv"

CMD ["/venv/bin/python3", "train.py"]
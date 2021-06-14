FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 vim git -y
RUN conda install torchvision
RUN pip install  opencv-python sklearn torchsummary

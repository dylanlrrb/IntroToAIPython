FROM nvidia/cuda:9.0-base

RUN mkdir -p /src

COPY ./requirements.txt /src

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

RUN pip install -r requirements.txt

RUN mkdir -p /~/.cache/torch/checkpoints

VOLUME /~/.cache/torch/checkpoints

WORKDIR /src

EXPOSE 8889

CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--no-browser", "--port=8889"]

# docker run --gpus all -p 8889:8889 -v (pwd)/final_project:/src -v (pwd)/cache:/~/.cache/torch/checkpoints --name jp-test cuda-test

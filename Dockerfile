FROM nvidia/cuda:9.0-base

ARG PYTHON_VERSION=python3.6

RUN apt-get update

RUN apt-get install software-properties-common -y

RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get update

RUN apt-get install $PYTHON_VERSION -y

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/$PYTHON_VERSION 2

RUN apt-get install -y python3-pip

RUN pip3 install --upgrade pip

RUN mkdir -p /src

COPY ./requirements.txt /src

RUN pip install -r src/requirements.txt

ARG NOTEBOOK_NAME=notebook.ipynb
# ARGs are only evaluated at build time
# store in an envirlnment variable so it can be used at run time
ENV NOTEBOOK_TITLE ${NOTEBOOK_NAME}

WORKDIR /src

# Ony need to Expose this port if you want to run the notebook with
# CMD jupyter notebook --no-browser --ip=0.0.0.0 --port=8889
# EXPOSE 8889

CMD jupyter nbconvert --to html --ExecutePreprocessor.timeout=3600 --execute ${NOTEBOOK_TITLE} --output=index.html
# CMD jupyter nbconvert --to html --ExecutePreprocessor.timeout=3600 --execute <notebook_to_execute>.ipynb --output=index.html

# docker build -t intro_ai .
# docker build -t <image-name> .

# docker run --gpus all --rm -p 8889:8889 -v (pwd)/src:/src -v (pwd)/container_cache/torch:/root/.cache/torch/checkpoints intro_ai
# docker run --gpus all --rm -p 8889:8889 -v (pwd)/<dir-containing-ipynb>:/src -v (pwd)/container_cache/torch:/root/.cache/torch/checkpoints <image-name>

# remove after stopping
# cache pre-trained models for use between containers

FROM nvidia/cuda:9.0-base

ENV PYTHON_VERSION python3.6

RUN apt-get update

RUN apt-get install software-properties-common -y

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update

RUN apt-get install $PYTHON_VERSION -y

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/$PYTHON_VERSION 2

RUN apt-get install -y python3-pip

RUN pip3 install --upgrade pip

RUN mkdir -p /src

COPY ./requirements.txt /src

RUN pip install -r src/requirements.txt

WORKDIR /src

EXPOSE 8889

CMD jupyter nbconvert --to html --ExecutePreprocessor.timeout=180000 --execute Image_Classifier_Project.ipynb --output=notebook.html
# CMD jupyter nbconvert --to html --ExecutePreprocessor.timeout=180000 --execute <notebook_to_execute>.ipynb --output=notebook.html

# docker build -t <image-name> .
# docker run --gpus all --rm -p 8889:8889 -v (pwd)/final_project:/src -v (pwd)/container_cache/torch:/root/.cache/torch/checkpoints <image-name>

# remove after stopping
# cache pre-trained models for use between containers

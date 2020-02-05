FROM nvidia/cuda:9.0-base

RUN mkdir -p /src

COPY ./requirements.txt /src

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

RUN pip install -r src/requirements.txt

WORKDIR /src

EXPOSE 8889

# CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--no-browser", "--port=8889"]
# docker run --gpus all --rm -p 8889:8889 -v (pwd)/final_project:/src -v (pwd)/container_cache/torch:/root/.cache/torch/checkpoints --name jp-test cuda-test
# remove after stopping
# cache pre-trained models for use between containers

CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "Image_Classifier_Project.ipynb", "--output=result.ipynb"]
# docker run --gpus all --rm -p 8889:8889 -v (pwd)/final_project:/src -v (pwd)/container_cache/torch:/root/.cache/torch/checkpoints --name result-test cuda-head

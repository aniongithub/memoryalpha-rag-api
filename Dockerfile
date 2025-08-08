FROM python:3.12-slim-bullseye AS devcontainer

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt-get -y install jq curl wget git

COPY ./requirements.txt /tmp/pip-tmp/
RUN pip install --no-cache-dir -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

WORKDIR /data
RUN wget https://github.com/aniongithub/memoryalpha-vectordb/releases/latest/download/enmemoryalpha_db.tar.gz &&\
    tar -xzf enmemoryalpha_db.tar.gz &&\
    rm enmemoryalpha_db.tar.gz &&\
    chmod -R 0777 /data

FROM devcontainer AS runtime

WORKDIR /workspace/memoryalpha-rag-api
COPY . /workspace/memoryalpha-rag-api

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
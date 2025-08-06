FROM ollama/ollama:latest

RUN apt-get update &&\
    apt-get install -y curl &&\
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/*

COPY ./bootstrap.sh /bootstrap.sh
RUN chmod +x /bootstrap.sh

EXPOSE 11434

ENTRYPOINT ["/bootstrap.sh"]
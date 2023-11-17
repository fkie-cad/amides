# syntax=docker/dockerfile:1

FROM python:3.11-slim-bullseye AS base

ARG GID=1001
ARG UID=1001

RUN apt-get update && apt-get upgrade -y && apt-get install -y jq && \
    addgroup --gid ${GID} docker-user && \
    adduser --uid ${UID} --gid ${GID} --disabled-password --gecos "" docker-user && \
    echo "docker-user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

ADD ./amides /home/docker-user/amides

RUN python -m venv /home/docker-user/amides/venv
ENV PATH="/home/docker-user/amides/venv/bin:$PATH"
RUN chown -R docker-user:docker-user /home/docker-user/amides

WORKDIR /home/docker-user/amides
USER docker-user
RUN pip install --upgrade pip && pip install -r requirements_dev.txt && pip install tox && pip install -e .
RUN chmod +x experiments.sh classification.sh rule_attribution.sh tainted_training.sh classification_other_types.sh








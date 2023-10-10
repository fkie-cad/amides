# syntax=docker/dockerfile:1

FROM python:3.11-slim-bullseye AS base

RUN apt-get update && apt-get upgrade -y && apt-get install -y jq

ADD ./amides /amides
WORKDIR /amides

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN python -m pip install --upgrade pip && pip install -r requirements.txt && pip install .
RUN chmod +x bin/results.sh








# To build the docker image, run the following command:
# docker build -t evol-aie4 .

FROM python:3.12-slim

WORKDIR /app

COPY . .
FROM nvidia/cuda:9.0-runtime
WORKDIR /app

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install pip --upgrade

COPY /inference/requirements.txt .
RUN pip3 install -r requirements.txt
RUN apt-get -y install libglib2.0

COPY /inference .

EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["/app/app.py"]
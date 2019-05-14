FROM nvidia/cuda:9.0-runtime
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    libglib2.0 \
  && rm -rf /var/lib/apt/lists/* 

RUN pip3 install pip --upgrade
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY /inference .

EXPOSE 5000
CMD ["python3", "/app/app.py"]
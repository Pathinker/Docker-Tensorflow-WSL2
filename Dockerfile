FROM tensorflow/tensorflow:2.17.0-gpu

WORKDIR /app

COPY . .

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
FROM python:3.9.5-slim-buster
RUN apt-get update \
    && apt-get -y install libpq-dev gcc 

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# CMD ["gunicorn","-w", "3", "--thread=4","--bind", "0.0.0.0:5000", "--timeout","600","--log-file", "-", "run:app"]
ENTRYPOINT ["python"]
CMD ["main.py"]
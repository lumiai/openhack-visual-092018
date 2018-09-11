FROM continuumio/anaconda3:5.2.0

# We copy just the requirements.txt first to leverage Docker cache
RUN conda install keras

COPY . /app

WORKDIR /app

ENTRYPOINT ["python"]
CMD ["app.py"]
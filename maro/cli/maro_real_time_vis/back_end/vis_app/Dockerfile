FROM python:3.6.7

RUN mkdir -p /home/project/vis_app
WORKDIR /home/project/vis_app
COPY requirements.txt /home/project/vis_app
RUN pip install --no-cache-dir -r requirements.txt

COPY . /home/project/vis_app
EXPOSE 5000
EXPOSE 9000

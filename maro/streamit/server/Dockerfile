FROM node:15-alpine

WORKDIR /maro_vis

RUN npm install -g http-server

CMD ["http-server", "/maro_vis", "-p", "9988", "--cors"]
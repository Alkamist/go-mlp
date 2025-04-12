FROM golang:1.24.2

WORKDIR /app

COPY ./go.mod ./

RUN mkdir -p machine_learning
RUN mkdir -p server
RUN mkdir -p models

COPY ./machine_learning/ ./machine_learning/
COPY ./server/ ./server/

RUN go build -o main ./server/server.go

EXPOSE 8080

CMD ["./main"]
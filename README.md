## What is this?

This repository hosts a simple Multilayer Perceptron (MLP) written in pure [Go](https://go.dev/), adapted from my example written in [Odin](https://odin-lang.org/) in [this repository](https://github.com/Alkamist/odin_machine_learning).

The MLP can be trained from scratch to classify handwritten digits using the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database), and then containerized via [Kubernetes](https://kubernetes.io) as a service that accepts input and responds with output through HTTP.

## Training the model

First, navigate to the `train` folder.

If you unzip mnist.zip, you should get two csv files, which are the training and validation sets of the MNIST dataset. 

Once you have the training and validation sets as csv files, you can run `go build train.go`, and then run the executable to train the MLP to classify handwritten digits.

The MLP should be able to reach around 97.5% validation set accuracy in around 15-20 epochs in several minutes, after which the weights and biases will be saved to a `model.json` file.

## Running as a service on Kubernetes

The following has been tested on [Rancher Desktop](https://rancherdesktop.io/).

Once you have trained a model, you can move the `model.json` into the `models` folder to be used by the server.

Navigate back to the root directory of the repository.

You will need to make a configmap so the server has access to the model file from within its container.

`kubectl create configmap model-config --from-file=model.json=./models/model.json`

Then you will need to create a docker image for digit classifier.

`docker build -t digit-classifier:latest .`

Once you have successfully made a docker image, you can then deploy it on Kubernetes.

`kubectl apply -f deployment.yaml -f service.yaml`

## Testing the service

After you have the service up and running, you can test it in the `client` folder.

In this folder, there is an image `number.png`, which you can edit and hand-draw a digit of your choosing.

If you run `go run client.go number.png`, it will send this image to the server so the hosted MLP can classify which digit you drew and respond back.

All of the MNIST dataset digits are centered, so try to keep your drawing mostly in the center for the highest accuracy.
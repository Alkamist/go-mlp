This is a simple Multilayer Perceptron written in pure [Go](https://go.dev/), adapted from my example written in [Odin](https://odin-lang.org/) in [this repository](https://github.com/Alkamist/odin_machine_learning).

If you unzip mnist.zip, you should get two csv files, which are the training and validation sets of the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database). 

The Mnist dataset is a collection of hand-drawn digits, each paired with the integer that the hand-drawn digit represents.

main.go will train a small Multilayer Perceptron on the training set, and validate it on the validation set every epoch.

The Multilayer Perceptron implementation is very simple and single threaded, so it isn't very fast, but it should be able to reach around 97.5% validation set accuracy in around 15 epochs in several minutes.
package main

import (
	"os"
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"encoding/csv"
	ml "rancher-test/machine_learning"
)

func main() {
	trainingSet   := newMnistData("./mnist_train.csv", 60000)
	validationSet := newMnistData("./mnist_test.csv",  10000)

	model := &ml.Mlp{}
	model.Init(_mnistImageSize, 100, _mnistClasses)

	// The order of the training set
	order := make([]int, len(trainingSet.output))
	for i := range order {
		order[i] = i
	}

	for epoch := range 55 {
		// Shuffle the training set
		rand.Shuffle(len(order), func(i, j int) {
			order[i], order[j] = order[j], order[i]
		})

		// Train on the training set
		for s := range len(trainingSet.output) {
			model.Learn(trainingSet.input[s][:], trainingSet.output[s][:])
		}

		// Validate with the validation set
		score := 0
		for s := range len(validationSet.output) {
			output := model.Forward(validationSet.input[s][:])
			if ml.Argmax(output[:]) == ml.Argmax(validationSet.output[s][:]) {
				score += 1
			}
		}

		accuracy := float32(score) / float32(len(validationSet.output))
		fmt.Printf("%v Validation set accuracy: %.2f%%\n", epoch, 100.0 * accuracy)

		if accuracy > 0.975 {
			break
		}
	}

	model.Save("model.json")
}

const _mnistImageSize = 784
const _mnistClasses   = 10

type MnistData struct {
	input  [][_mnistImageSize]float32
	output [][_mnistClasses]float32
}

func newMnistData(fileName string, size int) *MnistData {
	d := &MnistData{}

	fileData, err := os.ReadFile(fileName)
	if err != nil {
		fmt.Println("Failed to load mnist data from ", fileName)
		return nil
	}

	reader := csv.NewReader(strings.NewReader(string(fileData)))

	_, _ = reader.Read()

	d.input  = make([][_mnistImageSize]float32, size)
	d.output = make([][_mnistClasses]float32,   size)

	for n := range size {
		values, err := reader.Read()
		if err != nil {
			break
		}

		yInt, _ := strconv.ParseInt(values[0], 10, 64)
		d.output[n][yInt] = 1

		for i := range _mnistImageSize {
			valueInt, _ := strconv.ParseInt(values[i + 1], 10, 64)
			d.input[n][i] = float32(valueInt) / 255.0
		}
	}

	return d
}
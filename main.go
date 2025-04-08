package main

import (
	"os"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"encoding/csv"
)

func main() {
	trainingSet   := newMnistData("./mnist_train.csv", 60000)
	validationSet := newMnistData("./mnist_test.csv",  10000)

	model := newMlp(_mnistImageSize, 100, _mnistClasses)

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
			model.learn(trainingSet.input[s][:], trainingSet.output[s][:])
		}

		// Validate with the validation set
		score := 0
		for s := range len(validationSet.output) {
			output := model.forward(validationSet.input[s][:])
			if argmax(output[:]) == argmax(validationSet.output[s][:]) {
				score += 1
			}
		}
		fmt.Printf("%v Validation set accuracy: %.2f%%\n", epoch, 100.0 * float32(score) / float32(len(validationSet.output)))
	}
}

func sigmoidValue(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

func sigmoid(x []float32) []float32 {
	res := make([]float32, len(x))
	for i := range x {
		res[i] = sigmoidValue(x[i])
	}
	return res
}

func sigmoidDerivative(x float32) float32 {
	return sigmoidValue(x) * (1.0 - sigmoidValue(x))
}

func softmax(x []float32) []float32 {
	res := make([]float32, len(x))

	max_value := float32(-math.MaxFloat32)
	for i := range x {
		if x[i] > max_value {
			max_value = x[i]
		}
	}
	sum := float32(0.0)

	for i := range x {
		res[i] = float32(math.Exp(float64(x[i] - max_value)))
		sum += res[i]
	}
	for i := range res {
		res[i] /= sum
	}

	return res
}

func argmax(slice []float32) int {
	maxIndex := 0
	maxValue := slice[0]
	for i := 1; i < len(slice); i++ {
		if slice[i] > maxValue {
			maxValue = slice[i]
			maxIndex = i
		}
	}
	return maxIndex
}

// Parameters need 3 extra floating points for training
type Parameter struct {
	value    float32
	gradient float32
	adamM    float32
	adamV    float32
}

func (p *Parameter) update(timestep, batchSize int, learningRate, beta1, beta2, epsilon float32) {
	gradient := p.gradient / float32(batchSize)
	p.adamM = beta1 * p.adamM + (1 - beta1) * gradient
	p.adamV = beta2 * p.adamV + (1 - beta2) * gradient * gradient
	mCorrected := float64(p.adamM) / (1 - math.Pow(float64(beta1), float64(timestep)))
	vCorrected := float64(p.adamV) / (1 - math.Pow(float64(beta2), float64(timestep)))
	p.value -= float32(float64(learningRate) * mCorrected / (math.Sqrt(float64(vCorrected)) + float64(epsilon)))
	p.gradient = 0
}

type linear struct {
	weights [][]Parameter
	biases  []Parameter
}

func newLinear(inputSize, outputSize int) *linear {
	l := &linear{}

	l.weights = make([][]Parameter, outputSize)
	for o := range outputSize {
		l.weights[o] = make([]Parameter, inputSize)
	}
	l.biases = make([]Parameter, outputSize)

	scale := math.Sqrt(2.0 / float64(inputSize))
	for o := range outputSize {
		for i := range inputSize {
			l.weights[o][i].value = float32(rand.NormFloat64() * scale)
		}
	}

	return l
}

func (l *linear) forward(input []float32) []float32 {
	outputSize := len(l.biases)

	res := make([]float32, outputSize)

	for o := range outputSize {
		res[o] = l.biases[o].value
		for i := range input {
			res[o] += l.weights[o][i].value * input[i]
		}
	}

	return res
}

type mlp struct {
	timestep     int
	batchCounter int
	layer0       *linear
	layer1       *linear

	batchSize    int
	learningRate float32
	beta1        float32
	beta2        float32
	epsilon      float32
}

func newMlp(inputSize, hiddenSize, outputSize int) *mlp {
	m := &mlp{}

	m.layer0 = newLinear(inputSize, hiddenSize)
	m.layer1 = newLinear(hiddenSize, outputSize)

	m.batchSize    = 100
	m.learningRate = 0.001
	m.beta1        = 0.9
	m.beta2        = 0.999
	m.epsilon      = 1e-8

	return m
}

func (m *mlp) forward(input []float32) []float32 {
	hidden := sigmoid(m.layer0.forward(input))
	res    := softmax(m.layer1.forward(hidden))
	return res
}

func (m *mlp) learn(input, target []float32) {
	// Forward
	hidden           := m.layer0.forward(input)
	hidden_activated := sigmoid(hidden)
	output           := softmax(m.layer1.forward(hidden_activated))

	inputSize  := len(input)
	outputSize := len(output)
	hiddenSize := len(hidden)

	// Backward (gradient accumulation)
	deltas := make([]float32, outputSize)
	for o := range outputSize {
		deltas[o] = (output[o] - target[o]) * sigmoidDerivative(output[o])
		for h := range hiddenSize {
			m.layer1.weights[o][h].gradient += deltas[o] * hidden_activated[h]
		}
		m.layer1.biases[o].gradient += deltas[o]
	}
	for h := range hiddenSize {
		d_cost_o := float32(0.0)
		for o := range outputSize {
			d_cost_o += deltas[o] * m.layer1.weights[o][h].value
		}
		delta := d_cost_o * sigmoidDerivative(hidden[h])
		for i := range inputSize {
			m.layer0.weights[h][i].gradient += delta * input[i]
		}
		m.layer0.biases[h].gradient += delta
	}

	// Update parameters with accumulated gradients in batches
	m.batchCounter += 1
	if m.batchCounter > m.batchSize {
		m.timestep += 1
		for h := range hiddenSize {
			for i := range inputSize {
				m.layer0.weights[h][i].update(m.timestep, m.batchSize, m.learningRate, m.beta1, m.beta2, m.epsilon)
			}
			m.layer0.biases[h].update(m.timestep, m.batchSize, m.learningRate, m.beta1, m.beta2, m.epsilon)
		}
		for o := range outputSize {
			for h := range hiddenSize {
				m.layer1.weights[o][h].update(m.timestep, m.batchSize, m.learningRate, m.beta1, m.beta2, m.epsilon)
			}
			m.layer1.biases[o].update(m.timestep, m.batchSize, m.learningRate, m.beta1, m.beta2, m.epsilon)
		}
		m.batchCounter = 0
	}
}

const _mnistImageSize = 784
const _mnistClasses   = 10

type mnistData struct {
	input  [][_mnistImageSize]float32
	output [][_mnistClasses]float32
}

func newMnistData(fileName string, size int) *mnistData {
	d := &mnistData{}

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
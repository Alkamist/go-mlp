package machine_learning

import (
	"os"
	"fmt"
	"math"
	"math/rand"
	"encoding/json"
)

func sigmoidValue(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

func Sigmoid(x []float32) []float32 {
	res := make([]float32, len(x))
	for i := range x {
		res[i] = sigmoidValue(x[i])
	}
	return res
}

func SigmoidDerivative(x float32) float32 {
	return sigmoidValue(x) * (1.0 - sigmoidValue(x))
}

func Softmax(x []float32) []float32 {
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

func Argmax(slice []float32) int {
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
	Value    float32
	Gradient float32
	AdamM    float32
	AdamV    float32
}

func (p *Parameter) Update(timestep, batchSize int, learningRate, beta1, beta2, epsilon float32) {
	gradient := p.Gradient / float32(batchSize)
	p.AdamM = beta1 * p.AdamM + (1 - beta1) * gradient
	p.AdamV = beta2 * p.AdamV + (1 - beta2) * gradient * gradient
	mCorrected := float64(p.AdamM) / (1 - math.Pow(float64(beta1), float64(timestep)))
	vCorrected := float64(p.AdamV) / (1 - math.Pow(float64(beta2), float64(timestep)))
	p.Value -= float32(float64(learningRate) * mCorrected / (math.Sqrt(float64(vCorrected)) + float64(epsilon)))
	p.Gradient = 0
}

type LinearCheckpoint struct {
	Weights []float32 `json:"weights"`
	Biases  []float32 `json:"biases"`
}

type Linear struct {
	Weights [][]Parameter
	Biases  []Parameter
}

func (l *Linear) Init(inputSize, outputSize int) {
	l.Weights = make([][]Parameter, outputSize)
	for o := range outputSize {
		l.Weights[o] = make([]Parameter, inputSize)
	}
	l.Biases = make([]Parameter, outputSize)

	scale := math.Sqrt(2.0 / float64(inputSize))
	for o := range outputSize {
		for i := range inputSize {
			l.Weights[o][i].Value = float32(rand.NormFloat64() * scale)
		}
	}
}

func (l *Linear) InputSize() int {
	return len(l.Weights[0])
}

func (l *Linear) OutputSize() int {
	return len(l.Biases)
}

func (l *Linear) Forward(input []float32) []float32 {
	outputSize := l.OutputSize()

	res := make([]float32, outputSize)

	for o := range outputSize {
		res[o] = l.Biases[o].Value
		for i := range input {
			res[o] += l.Weights[o][i].Value * input[i]
		}
	}

	return res
}

func (l *Linear) Checkpoint() LinearCheckpoint {
	checkpoint := LinearCheckpoint{}

	inputSize  := l.InputSize()
	outputSize := l.OutputSize()

	checkpoint.Weights = make([]float32, inputSize * outputSize)
	checkpoint.Biases  = make([]float32, outputSize)

	for o := range outputSize {
		for i := range inputSize {
			checkpoint.Weights[o * inputSize + i] = l.Weights[o][i].Value
		}
		checkpoint.Biases[o] = l.Biases[o].Value
	}

	return checkpoint
}

func (l *Linear) LoadCheckpoint(checkpoint LinearCheckpoint) {
	inputSize  := l.InputSize()
	outputSize := l.OutputSize()

	for o := range outputSize {
		for i := range inputSize {
			l.Weights[o][i].Value = checkpoint.Weights[o * inputSize + i]
		}
		l.Biases[o].Value = checkpoint.Biases[o]
	}
}

type MlpCheckpoint struct {
	Layer0Checkpoint LinearCheckpoint
	Layer1Checkpoint LinearCheckpoint
}

type Mlp struct {
	Layer0 Linear
	Layer1 Linear

	BatchSize    int
	LearningRate float32
	Beta1        float32
	Beta2        float32
	Epsilon      float32

	timestep     int
	batchCounter int
}

func (m *Mlp) Init(inputSize, hiddenSize, outputSize int) {
	m.Layer0.Init(inputSize, hiddenSize)
	m.Layer1.Init(hiddenSize, outputSize)

	m.BatchSize    = 100
	m.LearningRate = 0.001
	m.Beta1        = 0.9
	m.Beta2        = 0.999
	m.Epsilon      = 1e-8
}

func (m *Mlp) Forward(input []float32) []float32 {
	hidden := Sigmoid(m.Layer0.Forward(input))
	res    := Softmax(m.Layer1.Forward(hidden))
	return res
}

func (m *Mlp) Learn(input, target []float32) {
	// Forward
	hidden           := m.Layer0.Forward(input)
	hidden_activated := Sigmoid(hidden)
	output           := Softmax(m.Layer1.Forward(hidden_activated))

	inputSize  := len(input)
	outputSize := len(output)
	hiddenSize := len(hidden)

	// Backward (gradient accumulation)
	deltas := make([]float32, outputSize)
	for o := range outputSize {
		deltas[o] = (output[o] - target[o]) * SigmoidDerivative(output[o])
		for h := range hiddenSize {
			m.Layer1.Weights[o][h].Gradient += deltas[o] * hidden_activated[h]
		}
		m.Layer1.Biases[o].Gradient += deltas[o]
	}
	for h := range hiddenSize {
		d_cost_o := float32(0.0)
		for o := range outputSize {
			d_cost_o += deltas[o] * m.Layer1.Weights[o][h].Value
		}
		delta := d_cost_o * SigmoidDerivative(hidden[h])
		for i := range inputSize {
			m.Layer0.Weights[h][i].Gradient += delta * input[i]
		}
		m.Layer0.Biases[h].Gradient += delta
	}

	// Update parameters with accumulated gradients in batches
	m.batchCounter += 1
	if m.batchCounter > m.BatchSize {
		m.timestep += 1
		for h := range hiddenSize {
			for i := range inputSize {
				m.Layer0.Weights[h][i].Update(m.timestep, m.BatchSize, m.LearningRate, m.Beta1, m.Beta2, m.Epsilon)
			}
			m.Layer0.Biases[h].Update(m.timestep, m.BatchSize, m.LearningRate, m.Beta1, m.Beta2, m.Epsilon)
		}
		for o := range outputSize {
			for h := range hiddenSize {
				m.Layer1.Weights[o][h].Update(m.timestep, m.BatchSize, m.LearningRate, m.Beta1, m.Beta2, m.Epsilon)
			}
			m.Layer1.Biases[o].Update(m.timestep, m.BatchSize, m.LearningRate, m.Beta1, m.Beta2, m.Epsilon)
		}
		m.batchCounter = 0
	}
}

func (m *Mlp) Checkpoint() MlpCheckpoint {
	return MlpCheckpoint{
		Layer0Checkpoint: m.Layer0.Checkpoint(),
		Layer1Checkpoint: m.Layer1.Checkpoint(),
	}
}

func (m *Mlp) LoadCheckpoint(checkpoint MlpCheckpoint) {
	m.Layer0.LoadCheckpoint(checkpoint.Layer0Checkpoint)
	m.Layer1.LoadCheckpoint(checkpoint.Layer1Checkpoint)
}

func (m *Mlp) Save(fileName string) {
	jsonData, err := json.Marshal(m.Checkpoint())
	if err != nil {
		fmt.Println("Failed to marshal model to JSON:", err)
		return
	}

	os.WriteFile(fileName, jsonData, 0644)
}

func (m *Mlp) Load(fileName string) {
	jsonData, err := os.ReadFile(fileName)
	if err != nil {
		fmt.Println("Failed to load model from JSON:", err)
		return
	}

	checkpoint := MlpCheckpoint{}
	err = json.Unmarshal(jsonData, &checkpoint)
	if err != nil {
		fmt.Println("Failed to unmarshal model from JSON:", err)
		return
	}

	m.LoadCheckpoint(checkpoint)
}
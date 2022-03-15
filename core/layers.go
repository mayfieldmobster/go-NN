package core

import "NN/activation_funcs"
import "math/rand"

type Layer struct {
	input_size, num_neurons int
	biases []float64
	weights [][]float64
	name, activation_func string
}

func (l Layer) forward(input []float64) []float64 {
	outputs := []float64{}
	for i := 0; i < len(l.weights); i++ {
		output := 0.0
		for j := 0; j < len(l.weights[i]); j++ {
			output += l.weights[i][j] * input[j]
			output += l.biases[i]
			outputs = append(outputs, output)
		}
	}
	return l.activation(outputs)
}

func (l Layer) backward() {
	
}

func (l Layer) activation(input []float64) []float64 {
	outputs := []float64{}
	switch l.activation_func {
	case "sigmoid":
		outputs = activation_funcs.Sigmoid(input)
	case "LeakyReLU":
		outputs = activation_funcs.LeakyReLU(input)
	case "tanh":
		outputs = activation_funcs.Tanh(input)
	case "ReLU":
		outputs = activation_funcs.ReLU(input)
	case "softmax":
		outputs = activation_funcs.softmax(input)
	case "":
		outputs = input
	}
	return outputs
}

func (l Layer) layer_init(){
	for i:=0; i < l.num_neurons; i++ {
		l.biases = append(l.biases, rand.Float64())
		l.weights = append(l.weights, []float64{})
		for j:=0; j < l.input_size; j++ {
			l.weights[i] = append(l.weights[i], rand.Float64())
		}
	}
}


package core

import "NN/activation_funcs"
import "NN/loss_functions"
import "math/rand"
import "errors"

type Layer struct {
	input_size, num_neurons int
	biases []float64
	weights [][]float64
	name, activation_func string
	outputs []float64
	gradients [][]float64
}

type model struct {
	layer1 Layer
	layer2 Layer
	layer3 Layer
	layer4 Layer
	loss_function string
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
	l.outputs = l.activation(outputs)
	return l.outputs
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


func (m model) forward(input []float64, y []float64) (float64, error) {
	outputs := []float64{}

	if m.layer1.input_size != len(input) {//check if input size is correct
		return 0.0, errors.New("Invalid input size for layer 1")
	}
	outputs = m.layer1.forward(input)//forward layer 1


	if m.layer2.name == "nil" {
		switch m.loss_function {
		case "cross_entropy":
			return loss_functions.CatergoricalCrossEntropy(outputs, y), nil
		case "mean_squared_error":
			return loss_functions.MeanSquaredError(outputs, y), nil
		case "sum_squared_residual":
			return loss_functions.SumSquaredRisiduals(outputs, y), nil
		}
	}
	if m.layer2.input_size != len(outputs) {//check if input size is correct
		return 0.0, errors.New("Invalid input size for layer 2")
	}
	outputs = m.layer2.forward(outputs)//forward layer 2


	if m.layer3.name == "nil" {
		switch m.loss_function {
		case "cross_entropy":
			return loss_functions.CatergoricalCrossEntropy(outputs, y), nil
		case "mean_squared_error":
			return loss_functions.MeanSquaredError(outputs, y), nil
		case "sum_squared_residual":
			return loss_functions.SumSquaredRisiduals(outputs, y), nil
		}
	}
	if m.layer3.input_size != len(outputs) {//check if input size is correct
		return 0.0, errors.New("Invalid input size for layer 3")
	}
	outputs = m.layer3.forward(outputs)


	if m.layer4.name == "nil" {
		switch m.loss_function {
		case "cross_entropy":
			return loss_functions.CatergoricalCrossEntropy(outputs, y), nil
		case "mean_squared_error":
			return loss_functions.MeanSquaredError(outputs, y), nil
		case "sum_squared_residual":
			return loss_functions.SumSquaredRisiduals(outputs, y), nil
		}
	}
	if m.layer4.input_size != len(outputs) {//check if input size is correct
		return 0.0, errors.New("Invalid input size for layer 4")
	}
	outputs = m.layer4.forward(outputs)//forward layer 4

	switch m.loss_function {
	case "cross_entropy":
		return loss_functions.CatergoricalCrossEntropy(outputs, y), nil
	case "mean_squared_error":
		return loss_functions.MeanSquaredError(outputs, y), nil
	case "sum_squared_residual":
		return loss_functions.SumSquaredRisiduals(outputs, y), nil
	}
	return 0.0, nil
}

func (m model) backward(y []float64){
	d_vals := []float64{}
	switch m.loss_function {
	case "cross_entropy":
		if m.layer4.name != "nil" {
			d_vals = loss_functions.CatergoricalCrossEntropy_derivative(m.layer4.outputs, y)
		}else if m.layer3.name != "nil" {
			d_vals = loss_functions.CatergoricalCrossEntropy_derivative(m.layer3.outputs, y)
		}else if m.layer2.name != "nil" {
			d_vals = loss_functions.CatergoricalCrossEntropy_derivative(m.layer2.outputs, y)
		}
	case "mean_squared_error":
		if m.layer4.name != "nil" {
			d_vals = loss_functions.MeanSquaredError_derivative(m.layer4.outputs, y)
		}else if m.layer3.name != "nil" {
			d_vals = loss_functions.MeanSquaredError_derivative(m.layer3.outputs, y)
		}else if m.layer2.name != "nil" {
			d_vals = loss_functions.MeanSquaredError_derivative(m.layer2.outputs, y)
		}
	case "sum_squared_residual":
		if m.layer4.name != "nil" {
			d_vals = loss_functions.SumSquaredRisiduals_derivative(m.layer4.outputs, y)
		}else if m.layer3.name != "nil" {
			d_vals = loss_functions.SumSquaredRisiduals_derivative(m.layer3.outputs, y)
		}else if m.layer2.name != "nil" {
			d_vals = loss_functions.SumSquaredRisiduals_derivative(m.layer2.outputs, y)
		}
	}
	for i:=0; i < len(m.layer4.weights); i++ {
}
}

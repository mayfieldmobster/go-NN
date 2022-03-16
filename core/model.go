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
	activated_outputs []float64
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
	l.outputs = outputs
	l.activated_outputs = l.activation(outputs)
	return l.activated_outputs
}


func (l Layer) activation(input []float64) []float64 {
	outputs := []float64{}
	switch l.activation_func {
	case "sigmoid":
		outputs = activation_funcs.sigmoid(input)
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

func (l Layer) layer_init(input_node bool){
	for i:=0; i < l.num_neurons; i++ {
		if input_node {
			l.biases = append(l.biases, 0.0)
			l.weights = append(l.weights, []float64{})
		}else {
			l.biases = append(l.biases, rand.Float64())
			l.weights = append(l.weights, []float64{})
		}
		
		for j:=0; j < l.input_size; j++ {
			if input_node {
				l.weights[i] = append(l.weights[i], 1.0)
			}else {
				l.weights[i] = append(l.weights[i], rand.Float64())}
		}
	}
}

func array_multiply(arr1 []float64, arr2 []float64) []float64 {
	output := []float64{}
	for i := 0; i < len(arr1); i++ {
		output = append(output, arr1[i] * arr2[i])
	}
	return output
}

func sum(arr []float64) float64 {
	sum := 0.0
	for i := 0; i < len(arr); i++ {
		sum += arr[i]
	}
	return sum
}

func array_multiply_scalar(arr []float64, scalar float64) []float64 {
	output := []float64{}
	for i := 0; i < len(arr); i++ {
		output = append(output, arr[i] * scalar)
	}
	return output
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


func (m model) back(y []float64){
	d_vals := []float64{}
	switch m.loss_function {
	case "cross_entropy":
		if m.layer4.name != "nil" {
			d_vals = loss_functions.CatergoricalCrossEntropy_derivative(m.layer4.activated_outputs, y)
		}else if m.layer3.name != "nil" {
			d_vals = loss_functions.CatergoricalCrossEntropy_derivative(m.layer3.activated_outputs, y)
		}else if m.layer2.name != "nil" {
			d_vals = loss_functions.CatergoricalCrossEntropy_derivative(m.layer2.activated_outputs, y)
		}
	case "mean_squared_error":
		if m.layer4.name != "nil" {
			d_vals = loss_functions.MeanSquaredError_derivative(m.layer4.activated_outputs, y)
		}else if m.layer3.name != "nil" {
			d_vals = loss_functions.MeanSquaredError_derivative(m.layer3.activated_outputs, y)
		}else if m.layer2.name != "nil" {
			d_vals = loss_functions.MeanSquaredError_derivative(m.layer2.activated_outputs, y)
		}
	case "sum_squared_residual":
		if m.layer4.name != "nil" {
			d_vals = loss_functions.SumSquaredRisiduals_derivative(m.layer4.activated_outputs, y)
		}else if m.layer3.name != "nil" {
			d_vals = loss_functions.SumSquaredRisiduals_derivative(m.layer3.activated_outputs, y)
		}else if m.layer2.name != "nil" {
			d_vals = loss_functions.SumSquaredRisiduals_derivative(m.layer2.activated_outputs, y)
		}
	}
	switch m.layer4.activation_func {
	case "sigmoid":
		d_vals = array_multiply(activation_funcs.sigmoid_derivative(m.layer4.outputs), d_vals) 
	case "LeakyReLU":
		d_vals = array_multiply(activation_funcs.LeakyReLU_derivative(m.layer4.outputs), d_vals)
	case "tanh":
		d_vals = array_multiply(activation_funcs.Tanh_derivative(m.layer4.outputs), d_vals)
	case "ReLU":
		d_vals = array_multiply(activation_funcs.ReLU_derivative(m.layer4.outputs), d_vals)
	case "softmax":
		d_vals = array_multiply(activation_funcs.softmax_derivative(m.layer4.outputs), d_vals)
	case "":
		d_vals = d_vals
	}
	if m.layer4.name != "nil" {
		for i := 0; i < len(m.layer4.weights); i++ {
			m.layer4.gradients = append(m.layer4.gradients, []float64{})
			for j:=0; j < len(m.layer4.weights[i]); j++ {
				m.layer4.gradients[i] = append(m.layer4.gradients[i], d_vals[i]*m.layer3.activated_outputs[j]) 
			}
		}
	}

	if m.layer4.name != "nil" {
		d_vals := []float64{}
		for i := 0; i < len(m.layer4.weights); i++ {
			for j:=0; j < len(m.layer4.weights[i]); j++ {
				d_vals = append(d_vals, 0.0)
				break
			}
		}
		for k := 0; k < len(m.layer4.gradients); k++ {
			for l:=0; l < len(m.layer4.gradients[k]); l++ {
				d_vals[l] += m.layer4.gradients[k][l]

			}
		}
	}
	switch m.layer3.activation_func {
	case "sigmoid":
		d_vals = array_multiply(activation_funcs.sigmoid_derivative(m.layer3.outputs), d_vals) 
	case "LeakyReLU":
		d_vals = array_multiply(activation_funcs.LeakyReLU_derivative(m.layer3.outputs), d_vals)
	case "tanh":
		d_vals = array_multiply(activation_funcs.Tanh_derivative(m.layer3.outputs), d_vals)
	case "ReLU":
		d_vals = array_multiply(activation_funcs.ReLU_derivative(m.layer3.outputs), d_vals)
	case "softmax":
		d_vals = array_multiply(activation_funcs.softmax_derivative(m.layer3.outputs), d_vals)
	case "":
		d_vals = d_vals
	}

	if m.layer3.name != "nil" {
		for i := 0; i < len(m.layer3.weights); i++ {
			m.layer4.gradients = append(m.layer3.gradients, []float64{})
			for j:=0; j < len(m.layer3.weights[i]); j++ {
				m.layer3.gradients[i] = append(m.layer3.gradients[i], d_vals[i]*m.layer2.activated_outputs[j]) 
			}
		}
	}
	if m.layer3.name != "nil" {
		d_vals := []float64{}
		for i := 0; i < len(m.layer3.weights); i++ {
			for j:=0; j < len(m.layer3.weights[i]); j++ {
				d_vals = append(d_vals, 0.0)
				break
			}
		}
		for k := 0; k < len(m.layer3.gradients); k++ {
			for l:=0; l < len(m.layer3.gradients[k]); l++ {
				d_vals[l] += m.layer3.gradients[k][l]

			}
		}
	}

	switch m.layer2.activation_func {
	case "sigmoid":
		d_vals = array_multiply(activation_funcs.sigmoid_derivative(m.layer2.outputs), d_vals) 
	case "LeakyReLU":
		d_vals = array_multiply(activation_funcs.LeakyReLU_derivative(m.layer2.outputs), d_vals)
	case "tanh":
		d_vals = array_multiply(activation_funcs.Tanh_derivative(m.layer2.outputs), d_vals)
	case "ReLU":
		d_vals = array_multiply(activation_funcs.ReLU_derivative(m.layer2.outputs), d_vals)
	case "softmax":
		d_vals = array_multiply(activation_funcs.softmax_derivative(m.layer2.outputs), d_vals)
	case "":
		d_vals = d_vals
	}
	if m.layer2.name != "nil" {
		for i := 0; i < len(m.layer2.weights); i++ {
			m.layer4.gradients = append(m.layer2.gradients, []float64{})
			for j:=0; j < len(m.layer2.weights[i]); j++ {
				m.layer2.gradients[i] = append(m.layer2.gradients[i], d_vals[i]*m.layer1.activated_outputs[j]) 
			}
		}
	}
}

func (m model) weight_update(learning_rate float64) {
	if m.layer4.name != "nil" {
		for i := 0; i < len(m.layer4.weights); i++ {
			for j:=0; j < len(m.layer4.weights[i]); j++ {
				m.layer4.weights[i][j] += m.layer4.gradients[i][j]*learning_rate
			}
		}
		m.layer4.gradients = [][]float64{}
	}
	if m.layer3.name != "nil" {
		for i := 0; i < len(m.layer3.weights); i++ {
			for j:=0; j < len(m.layer3.weights[i]); j++ {
				m.layer3.weights[i][j] += m.layer3.gradients[i][j]*learning_rate
			}
		}
		m.layer3.gradients = [][]float64{}
	}
	if m.layer2.name != "nil" {
		for i := 0; i < len(m.layer2.weights); i++ {
			for j:=0; j < len(m.layer2.weights[i]); j++ {
				m.layer2.weights[i][j] += m.layer2.gradients[i][j]*learning_rate
			}
		}
		m.layer2.gradients = [][]float64{}
	}
}

func (m model) train(training_data [][]float64, labels [][]float64 ,learning_rate float64, epochs int) {
	for i := 0; i < epochs; i++ {
		for j := 0; j < len(training_data); j++ {
			m.forward(training_data[j], labels[j])
			m.back(labels[j])
			m.weight_update(learning_rate)
		}
	}
}

func one_hot(label int, num_classes int) []float64 {
	output := []float64{}
	for i := 0; i < num_classes; i++ {
		if i == label {
			output = append(output, 1.0)
		} else {
			output = append(output, 0.0)
		}
	}
	return output
}

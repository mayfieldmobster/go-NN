package core

import (
	"NN/activation_funcs"
	"NN/loss_functions"
	"errors"
	"fmt"
	"math/rand"
	"time"
	//"os"
)


type Layer struct {
	Input_size, Num_neurons int
	Weights [][]float64
	Name, Activation_func string
	Outputs []float64
	Activated_outputs []float64
	Gradients [][]float64
}

type Model struct {
	Layer1 *Layer
	Layer2 *Layer
	Layer3 *Layer
	Layer4 *Layer
	Loss_function string
}


func (l *Layer) Forward(input []float64) []float64 {
	outputs := []float64{}
	for i := 0; i < len(l.Weights); i++ {
		output := 0.0
		for j := 0; j < len(l.Weights[i]); j++ {
			output += l.Weights[i][j] * input[j]
		}
		outputs = append(outputs, output)
	}
	l.Outputs = outputs
	l.Activated_outputs = l.Activation(outputs)
	return l.Activated_outputs
}


func (l Layer) Activation(input []float64) []float64 {
	outputs := []float64{}
	switch l.Activation_func {
	case "sigmoid":
		outputs = activation_funcs.Sigmoid(input)
	case "LeakyReLU":
		outputs = activation_funcs.LeakyReLU(input)
	case "tanh":
		outputs = activation_funcs.Tanh(input)
	case "ReLU":
		outputs = activation_funcs.ReLU(input)
	case "softmax":
		outputs = activation_funcs.Softmax(input)
	case "":
		outputs = input
	}
	return outputs
}

func (l *Layer) Layer_init(input_node bool){
	rand.Seed(time.Now().UTC().UnixNano())
	for i:=0; i < l.Num_neurons; i++ {
		l.Weights = append(l.Weights, []float64{})
		for j:=0; j < l.Input_size; j++ {
			if input_node {
				l.Weights[i] = append(l.Weights[i], 1.0)
			}else {
				if rand.Intn(2) == 0 {
					l.Weights[i] = append(l.Weights[i], -rand.Float64()*0.1)
				}else {
					l.Weights[i] = append(l.Weights[i], rand.Float64()*0.1)
				}
			}
		}
	}
	fmt.Println(l.Weights)
}

func Array_multiply(arr1 []float64, arr2 []float64) []float64 {
	output := []float64{}
	for i := 0; i < len(arr1); i++ {
		output = append(output, arr1[i] * arr2[i])
	}
	return output
}

func Sum(arr []float64) float64 {
	sum := 0.0
	for i := 0; i < len(arr); i++ {
		sum += arr[i]
	}
	return sum
}

func Array_multiply_scalar(arr []float64, scalar float64) []float64 {
	output := []float64{}
	for i := 0; i < len(arr); i++ {
		output = append(output, arr[i] * scalar)
	}
	return output
}

func (m *Model) Forward(input []float64, y []float64) (float64, error) {
	outputs := []float64{}

	if m.Layer1.Input_size != len(input) {//check if input size is correct
		return 0.0, errors.New("invalid input size for layer 1")
	}
	outputs = m.Layer1.Forward(input)//forward layer 1


	if m.Layer2.Name == "nil" {
		switch m.Loss_function {
		case "cross_entropy":
			return loss_functions.CatergoricalCrossEntropy(outputs, y), nil
		case "mean_squared_error":
			return loss_functions.MeanSquaredError(outputs, y), nil
		case "sum_squared_residual":
			return loss_functions.SumSquaredRisiduals(outputs, y), nil
		}
	}
	if m.Layer2.Input_size != len(outputs) {//check if input size is correct
		return 0.0, errors.New("invalid input size for layer 2")
	}
	outputs = m.Layer2.Forward(outputs)//forward layer 2


	if m.Layer3.Name == "nil" {
		switch m.Loss_function {
		case "cross_entropy":
			return loss_functions.CatergoricalCrossEntropy(outputs, y), nil
		case "mean_squared_error":
			return loss_functions.MeanSquaredError(outputs, y), nil
		case "sum_squared_residual":
			return loss_functions.SumSquaredRisiduals(outputs, y), nil
		}
	}
	if m.Layer3.Input_size != len(outputs) {//check if input size is correct
		return 0.0, errors.New("invalid input size for layer 3")
	}
	outputs = m.Layer3.Forward(outputs)


	if m.Layer4.Name == "nil" {
		switch m.Loss_function {
		case "cross_entropy":
			return loss_functions.CatergoricalCrossEntropy(outputs, y), nil
		case "mean_squared_error":
			return loss_functions.MeanSquaredError(outputs, y), nil
		case "sum_squared_residual":
			return loss_functions.SumSquaredRisiduals(outputs, y), nil
		}
	}
	if m.Layer4.Input_size != len(outputs) {//check if input size is correct
		return 0.0, errors.New("invalid input size for layer 4")
	}
	outputs = m.Layer4.Forward(outputs)//forward layer 4
	switch m.Loss_function {
	case "cross_entropy":
		return loss_functions.CatergoricalCrossEntropy(outputs, y), nil
	case "mean_squared_error":
		return loss_functions.MeanSquaredError(outputs, y), nil
	case "sum_squared_residual":
		return loss_functions.SumSquaredRisiduals(outputs, y), nil
	}
	return 0.0, nil
}


func (m *Model) Back(input []float64, y []float64){
	//fmt.Println("")
	d_vals := []float64{}
	d_loss := []float64{}
	switch m.Loss_function {
	case "cross_entropy":
		if m.Layer4.Name != "nil" {
			d_loss = loss_functions.CatergoricalCrossEntropy_derivative(m.Layer4.Activated_outputs, y)
		}else if m.Layer3.Name != "nil" {
			d_loss = loss_functions.CatergoricalCrossEntropy_derivative(m.Layer3.Activated_outputs, y)
		}else if m.Layer2.Name != "nil" {
			d_loss = loss_functions.CatergoricalCrossEntropy_derivative(m.Layer2.Activated_outputs, y)
		}
	case "mean_squared_error":
		if m.Layer4.Name != "nil" {
			d_loss = loss_functions.MeanSquaredError_derivative(m.Layer4.Activated_outputs, y)
		}else if m.Layer3.Name != "nil" {
			d_loss = loss_functions.MeanSquaredError_derivative(m.Layer3.Activated_outputs, y)
		}else if m.Layer2.Name != "nil" {
			d_loss = loss_functions.MeanSquaredError_derivative(m.Layer2.Activated_outputs, y)
		}
	case "sum_squared_residual":
		if m.Layer4.Name != "nil" {
			d_loss = loss_functions.SumSquaredRisiduals_derivative(m.Layer4.Activated_outputs, y)
		}else if m.Layer3.Name != "nil" {
			d_loss = loss_functions.SumSquaredRisiduals_derivative(m.Layer3.Activated_outputs, y)
		}else if m.Layer2.Name != "nil" {
			d_loss = loss_functions.SumSquaredRisiduals_derivative(m.Layer2.Activated_outputs, y)
		}
	}
	//fmt.Println("d_loss: ", d_loss)
	
	switch m.Layer4.Activation_func {
	case "sigmoid":
		d_vals = Array_multiply(activation_funcs.Sigmoid_derivative(m.Layer4.Outputs), d_loss) 
	case "LeakyReLU":
		d_vals = Array_multiply(activation_funcs.LeakyReLU_derivative(m.Layer4.Outputs), d_loss)
	case "tanh":
		d_vals = Array_multiply(activation_funcs.Tanh_derivative(m.Layer4.Outputs), d_loss)
	case "ReLU":
		d_vals = Array_multiply(activation_funcs.ReLU_derivative(m.Layer4.Outputs), d_loss)
	case "softmax":
		d_vals = Array_multiply(activation_funcs.Softmax_derivative(m.Layer4.Outputs), d_loss)
	case "":
		d_vals = d_loss
	}
	//fmt.Println("d_vals AL1: ", d_vals)
	
	if m.Layer4.Name != "nil" {
		for i := 0; i < len(m.Layer4.Weights); i++ {
			m.Layer4.Gradients = append(m.Layer4.Gradients, []float64{})
			for j:=0; j < len(m.Layer4.Weights[i]); j++ {
				m.Layer4.Gradients[i] = append(m.Layer4.Gradients[i], d_vals[i]*m.Layer3.Activated_outputs[j]) 
			}
		}
	}

	if m.Layer4.Name != "nil" {
		new_d_vals := []float64{}
		
		for i := 0; i < len(m.Layer4.Weights[0]); i++ {
			new_d_val := 0.0
			for j:=0; j < len(m.Layer4.Weights); j++ {
				new_d_val += m.Layer4.Weights[j][i]*d_vals[j]
			}
			new_d_vals = append(new_d_vals, new_d_val)
		}
		d_vals = new_d_vals
	}
	//fmt.Println("d_vals WL1: ", d_vals)
	
	switch m.Layer3.Activation_func {
	case "sigmoid":
		d_vals = Array_multiply(activation_funcs.Sigmoid_derivative(m.Layer3.Outputs), d_vals) 
	case "LeakyReLU":
		d_vals = Array_multiply(activation_funcs.LeakyReLU_derivative(m.Layer3.Outputs), d_vals)
	case "tanh":
		d_vals = Array_multiply(activation_funcs.Tanh_derivative(m.Layer3.Outputs), d_vals)
	case "ReLU":
		d_vals = Array_multiply(activation_funcs.ReLU_derivative(m.Layer3.Outputs), d_vals)
	case "softmax":
		d_vals = Array_multiply(activation_funcs.Softmax_derivative(m.Layer3.Outputs), d_vals)
	}
	//fmt.Println("d_vals AL2: ", d_vals)
	if m.Layer3.Name != "nil" {
		for i := 0; i < len(m.Layer3.Weights); i++ {
			m.Layer3.Gradients = append(m.Layer3.Gradients, []float64{})
			for j:=0; j < len(m.Layer3.Weights[i]); j++ {
				m.Layer3.Gradients[i] = append(m.Layer3.Gradients[i], d_vals[i]*m.Layer2.Activated_outputs[j]) 
			}
		}
	}
	if m.Layer3.Name != "nil" {
		new_d_vals := []float64{}
		
		for i := 0; i < len(m.Layer3.Weights[0]); i++ {
			new_d_val := 0.0
			for j:=0; j < len(m.Layer3.Weights); j++ {
				new_d_val += m.Layer3.Weights[j][i]*d_vals[j]
			}
			new_d_vals = append(new_d_vals, new_d_val)
		}
		d_vals = new_d_vals
	}
	//fmt.Println("d_vals WL2: ", d_vals)
	switch m.Layer2.Activation_func {
	case "sigmoid":
		d_vals = Array_multiply(activation_funcs.Sigmoid_derivative(m.Layer2.Outputs), d_vals) 
	case "LeakyReLU":
		d_vals = Array_multiply(activation_funcs.LeakyReLU_derivative(m.Layer2.Outputs), d_vals)
	case "tanh":
		d_vals = Array_multiply(activation_funcs.Tanh_derivative(m.Layer2.Outputs), d_vals)
	case "ReLU":
		d_vals = Array_multiply(activation_funcs.ReLU_derivative(m.Layer2.Outputs), d_vals)
	case "softmax":
		d_vals = Array_multiply(activation_funcs.Softmax_derivative(m.Layer2.Outputs), d_vals)
	}
	//fmt.Println("d_vals AL3: ", d_vals)
	if m.Layer2.Name != "nil" {
		for i := 0; i < len(m.Layer2.Weights); i++ {
			m.Layer2.Gradients = append(m.Layer2.Gradients, []float64{})
			for j:=0; j < len(m.Layer2.Weights[i]); j++ {
				m.Layer2.Gradients[i] = append(m.Layer2.Gradients[i], d_vals[i]*m.Layer1.Activated_outputs[j]) 
			}
		}
	}
	if m.Layer2.Name != "nil" {
		new_d_vals := []float64{}
		
		for i := 0; i < len(m.Layer2.Weights[0]); i++ {
			new_d_val := 0.0
			for j:=0; j < len(m.Layer2.Weights); j++ {
				new_d_val += m.Layer2.Weights[j][i]*d_vals[j]
			}
			new_d_vals = append(new_d_vals, new_d_val)
		}
		d_vals = new_d_vals
	}
	//fmt.Println("d_vals WL3: ", d_vals)
	switch m.Layer1.Activation_func {
	case "sigmoid":
		d_vals = Array_multiply(activation_funcs.Sigmoid_derivative(m.Layer1.Outputs), d_vals) 
	case "LeakyReLU":
		d_vals = Array_multiply(activation_funcs.LeakyReLU_derivative(m.Layer1.Outputs), d_vals)
	case "tanh":
		d_vals = Array_multiply(activation_funcs.Tanh_derivative(m.Layer1.Outputs), d_vals)
	case "ReLU":
		d_vals = Array_multiply(activation_funcs.ReLU_derivative(m.Layer1.Outputs), d_vals)
	case "softmax":
		d_vals = Array_multiply(activation_funcs.Softmax_derivative(m.Layer1.Outputs), d_vals)
	}
	//fmt.Println("d_vals AL4: ", d_vals)
	if m.Layer1.Name != "nil" {
		for i := 0; i < len(m.Layer1.Weights); i++ {
			m.Layer1.Gradients = append(m.Layer1.Gradients, []float64{})
			for j:=0; j < len(m.Layer1.Weights[i]); j++ {
				m.Layer1.Gradients[i] = append(m.Layer1.Gradients[i], d_vals[i]*input[j]) 
			}
		}
	}
	/*
	if m.Layer1.Name != "nil" {
		new_d_vals := []float64{}
		
		for i := 0; i < len(m.Layer1.Weights[0]); i++ {
			new_d_val := 0.0
			for j:=0; j < len(m.Layer1.Weights); j++ {
				new_d_val += m.Layer1.Weights[j][i]*d_vals[j]
			}
			new_d_vals = append(new_d_vals, new_d_val)
		}
		d_vals = new_d_vals
	}
	*/

}

func (m Model) Weight_update(learning_rate float64) {
	fmt.Println("")
	if m.Layer4.Name != "nil" {
		for i := 0; i < len(m.Layer4.Weights); i++ {
			for j:=0; j < len(m.Layer4.Weights[i]); j++ {
				m.Layer4.Weights[i][j] += -m.Layer4.Gradients[i][j]*learning_rate
			}
		}
		fmt.Println(m.Layer4.Gradients)
		m.Layer4.Gradients = [][]float64{}
	}
	if m.Layer3.Name != "nil" {
		for i := 0; i < len(m.Layer3.Weights); i++ {
			for j:=0; j < len(m.Layer3.Weights[i]); j++ {
				m.Layer3.Weights[i][j] += -m.Layer3.Gradients[i][j]*learning_rate
			}
		}
		fmt.Println(m.Layer3.Gradients)
		m.Layer3.Gradients = [][]float64{}
	}
	if m.Layer2.Name != "nil" {
		for i := 0; i < len(m.Layer2.Weights); i++ {
			for j:=0; j < len(m.Layer2.Weights[i]); j++ {
				m.Layer2.Weights[i][j] += -m.Layer2.Gradients[i][j]*learning_rate
			}
		}
		fmt.Println(m.Layer2.Gradients)
		m.Layer2.Gradients = [][]float64{}
	}
	if m.Layer1.Name != "nil" {
		for i := 0; i < len(m.Layer1.Weights); i++ {
			for j:=0; j < len(m.Layer1.Weights[i]); j++ {
				m.Layer1.Weights[i][j] += -m.Layer1.Gradients[i][j]*learning_rate
			}
		}
		fmt.Println(m.Layer1.Gradients)
		m.Layer1.Gradients = [][]float64{}
		
	}
}

func (m Model) Train(training_data [][]float64, labels [][]float64 ,learning_rate float64, epochs int) {
	for i := 0; i < epochs; i++ {
		for j := 0; j < len(training_data); j++ {
			//fmt.Printf("\n\nEpoch: %d\n", i)
			m.Forward(training_data[j], labels[j])
			m.Back(training_data[j],labels[j])
			m.Weight_update(learning_rate)
			if j == 10000 {
				return
			}
		}
	}
}

func (m Model) Predict(input []float64) []float64 {
	m.Forward(input, []float64{})
	return m.Layer3.Activated_outputs
}

func (m Model) Test(test_data [][]float64, labels [][]float64, test_size int) float64 {
	correct := 0
	for i := 0; i < test_size; i++ {
		m.Forward(test_data[i], labels[i])
		fmt.Println(find_max(m.Layer4.Activated_outputs), find_max(labels[i]))
		if find_max(m.Layer4.Activated_outputs) == find_max(labels[i]) {
			correct++
		}
	}	
	return (float64(correct) / float64(len(test_data)))*100
}

func Array2D_to_1D( arr [][]float64) []float64 {
	output := []float64{}
	for i := 0; i < len(arr); i++ {
		for j := 0; j < len(arr[i]); j++ {
			output = append(output, arr[j][i])
		}
	}
	return output
}


func One_hot(label int, num_classes int) []float64 {
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

func Decrease_size(arr []float64) []float64 {
	output := []float64{}
	for i := 0; i < len(arr); i++ {
		output = append(output, arr[i]/255.0)
	}
	return output
}

func Unit8_to_float64(arr [][]uint8) [][]float64 {
	output := [][]float64{}
	for i := 0; i < len(arr); i++ {
		output = append(output, []float64{})
		for j := 0; j < len(arr[i]); j++ {
			output[i] = append(output[i], float64(arr[i][j]))
		}
	}
	return output
}

func find_max(arr []float64) int {
	max := 0
	for i := 0; i < len(arr); i++ {
		if arr[i] > arr[max] {
			max = i
		}
	}
	return max
}
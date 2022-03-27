package main

import (
	"NN/core"
	"NN/mnist"
	"fmt"
	"time"
)

func main() {
	data,err  := mnist.ReadTrainSet("c:/Users/Chris/go/src/NN/data")
	if err != nil {
		return
	}
	inputs := [][]float64{}
	labels := [][]float64{} 
	for _,data := range data.Data{
		inputs = append(inputs,core.Decrease_size(core.Array2D_to_1D(core.Unit8_to_float64(data.Image))))
		labels = append(labels,core.One_hot(data.Digit,10))
	}

	input_layer := &core.Layer{
		Input_size:784,
		Num_neurons:32,
		Weights:[][]float64{},
		Name:"input_layer",
		Outputs:[]float64{},
		Activated_outputs:[]float64{},
		Activation_func:"sigmoid",
		Gradients:[]float64{},
	}
	hidden_layer1 := &core.Layer{
		Input_size:32,
		Num_neurons:10,
		Weights:[][]float64{},
		Name:"hidden_layer1",
		Outputs:[]float64{},
		Activated_outputs:[]float64{},
		Activation_func:"sigmoid",
		Gradients:[]float64{},
	}
	hidden_layer2 := &core.Layer{
		Input_size:10,
		Num_neurons:10,
		Weights:[][]float64{},
		Name:"hidden_layer2",
		Outputs:[]float64{},
		Activated_outputs:[]float64{},
		Activation_func:"softmax",
		Gradients:[]float64{},
	}
	output_layer := &core.Layer{
		Input_size:10,
		Num_neurons:10,
		Weights:[][]float64{},
		Name:"output_layer",
		Outputs:[]float64{},
		Activated_outputs:[]float64{},
		Activation_func:"softmax",
		Gradients:[]float64{},
	}
	input_layer.Layer_init(false)
	time.Sleep(1 * time.Second)
	hidden_layer1.Layer_init(false)
	time.Sleep(1 * time.Second)
	hidden_layer2.Layer_init(false)
	time.Sleep(1 * time.Second)
	output_layer.Layer_init(false)
	
	model := &core.Model{
		Layer1:input_layer,
		Layer2:hidden_layer1,
		Layer3:hidden_layer2,
		Layer4:output_layer,
		Loss_function:"cross_entropy",
	}
	//fmt.Println(model.Layer4.Weights)
	//fmt.Println(len(model.Layer2.Weights))
	//fmt.Println(len(model.Layer2.Weights[0]))
	//fmt.Println(output_layer.Weights)
	//fmt.Println("Accuracy:",model.Test(inputs, labels, 1000), "%")
	model.Train(inputs, labels, 0.0001, 10)
	fmt.Println("Accuracy:",model.Test(inputs, labels, 1000), "%")
	//fmt.Println(model.Layer1.Weights)
	//fmt.Println(model.Layer2.Weights)
	//fmt.Println(model.Layer3.Weights)
	fmt.Println(model.Layer4.Weights)

}

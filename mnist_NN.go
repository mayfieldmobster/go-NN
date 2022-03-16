package main

import (
	"NN/core"
	"NN/loss_functions"
	"NN/mnist"
	"fmt"
)
func main() {
	data,err  := mnist.ReadTrainSet("c:/Users/Chris/go/src/NN/data")
	if err == nil {
		return
	}
	inputs := [][]float64{}
	labels := []float64{} 
	for _,data := range data.Data{
		inputs = append(inputs,core.decrease_size(core._2D_to_1D(core.unit8_to_float64(data.Image))))
			labels = append(labels,float64(data.Digit))
	}
	input_layer := core.Layer{
		input_size:784,
		num_neurons:32,
		weights:[][]float64{},
		name:"input_layer",
		outpusts:[]float64{},
		actiated_outputs:[]float64{},
		activation_function:"sigmoid",
		gradients:[][]float64{},
	}
	hidden_layer1 := core.Layer{
		input_size:32,
		num_neurons:10,
		weights:[][]float64{},
		name:"hidden_layer1",
		outpusts:[]float64{},
		actiated_outputs:[]float64{},
		activation_function:"",
		gradients:[][]float64{},
	}
	hidden_layer2 := core.Layer{
		input_size:10,
		num_neurons:10,
		weights:[][]float64{},
		name:"hidden_layer2",
		outpusts:[]float64{},
		actiated_outputs:[]float64{},
		activation_function:"sigmoid",
		gradients:[][]float64{},
	}
	output_layer := core.Layer{
		input_size:10,
		num_neurons:10,
		weights:[][]float64{},
		name:"output_layer",
		outpusts:[]float64{},
		actiated_outputs:[]float64{},
		activation_function:"softmax",
		gradients:[][]float64{},
	}
	input_layer.layer_init(true)
	hidden_layer1.layer_init(false)
	hidden_layer2.layer_init(false)
	output_layer.layer_init(false)
	
	model := core.Model{
		Layer1:input_layer,
		Layer2:hidden_layer1,
		Layer3:hidden_layer2,
		Layer4:output_layer,
		loss_function:"catergorical_cross_entropy",
	}
	model.train()
}

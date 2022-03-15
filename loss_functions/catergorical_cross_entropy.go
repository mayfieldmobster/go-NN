package loss_functions

import "math"

func CatergoricalCrossEntropy(arr []float64, y []float64) float64 {
	y = []float64{}
	for i := 0; i < len(arr); i++ {
		y = append(y, -y[i]*math.Log(arr[i])-(1-y[i])*math.Log(1-arr[i]))
	}
	total_loss := 0.0
	for _,value:= range y {
		total_loss += value
	}
	return total_loss
}

func CatergoricalCrossEntropy_derivative(arr []float64, y []float64) []float64 {
	y = []float64{}
	for i := 0; i < len(arr); i++ {
		y = append(y, -y[i]/arr[i]+(1-y[i])/(1-arr[i]))
	}
	return y
}

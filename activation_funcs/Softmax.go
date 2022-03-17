package activation_funcs

import "math"

func Softmax(arr []float64) []float64 {
	sum := 0.0
	for i := 0; i < len(arr); i++ {
		sum += math.Exp(arr[i])
	}
	y := []float64{}
	for i := 0; i < len(arr); i++ {
		y = append(y, math.Exp(arr[i])/sum)
	}
	return y
}

func Softmax_derivative(arr []float64) []float64 {
	y := []float64{}
	for i := 0; i < len(arr); i++ {
		y = append(y, Softmax(arr)[i]*(1-Softmax(arr)[i]))
	}
	return y
}
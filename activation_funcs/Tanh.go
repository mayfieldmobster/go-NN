package activation_funcs

import "math"

func Tanh(arr []float64 ) []float64 {
	y := []float64{}
	for i := 0; i < len(arr); i++ {
		y = append(y, math.Tanh(arr[i]))
	}
	return y 
}

func Tanh_derivative(arr []float64 ) []float64 {
	y := []float64{}
	for i := 0; i < len(arr); i++ {
		y = append(y, 1-math.Pow(math.Tanh(arr[i]),2))
	}
	return y 
}

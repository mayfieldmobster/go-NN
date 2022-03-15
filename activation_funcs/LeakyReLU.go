package activation_funcs

import "math"

func LeakyReLU(arr []float64 ) []float64 {
	y := []float64{}
	for i := 0; i < len(arr); i++ {
	y = append(y, math.Max(0.1*arr[i], arr[i]))
	}
	return y 
}

func LeakyReLU_derivative(arr []float64 ) []float64 {
	y := []float64{}
	for i := 0; i < len(arr); i++ {
	if arr[i] > 0 {
		y = append(y, 1)
	} else {
		y = append(y, 0.1)
	}
	}
	return y 
}
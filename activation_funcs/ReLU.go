package activation_funcs

import "math"

func ReLU(arr[]float64 ) []float64 {
	y := []float64{}
	for i := 0; i < len(arr); i++ {
	y = append(y, math.Max(0, arr[i]))
	}
	return y 
}

func ReLU_derivative(arr[]float64 ) []float64 {
	y := []float64{}
	for i := 0; i < len(arr); i++ {
	if arr[i] > 0 {
		y = append(y, 1)
	} else {
		y = append(y, 0)
	}
	}
	return y 
}
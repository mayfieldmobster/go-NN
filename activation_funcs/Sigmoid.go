package activation_funcs

import "math"

func sigmoid(arr[]float64 ) []float64 {
	y := []float64{}
	for i := 0; i < len(arr); i++ {
		y = append(y, 1/(1+math.Exp(-arr[i])))
	}
	return y 
}

func sigmoid_derivative(arr[]float64 ) []float64 {
	y := []float64{}
	for i := 0; i < len(arr); i++ {
		y = append(y, (1/(1+math.Exp(-arr[i])))*(1-(1/(1+math.Exp(-arr[i])))))
	}
	return y 
}
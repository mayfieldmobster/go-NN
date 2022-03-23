package loss_functions

import "math"

func CatergoricalCrossEntropy(arr []float64, labels []float64) float64 {
	loss := 0.0
	for i := 0; i < len(arr); i++ {
		loss += labels[i]*math.Log2(arr[i])
	}
	return -loss
}

func CatergoricalCrossEntropy_derivative(arr []float64, labels []float64) []float64 {
	d_loss := []float64{}
	for i := 0; i < len(arr); i++ {
		d_loss = append(d_loss, -(labels[i]/arr[i]))
		if d_loss[i] == -0.0 {
			d_loss[i] = 0.0
		}
	}
	return d_loss
}

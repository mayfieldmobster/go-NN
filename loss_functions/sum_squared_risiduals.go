package loss_functions

import "math"

func SumSquaredRisiduals(arr []float64, y []float64) float64 {
	y = []float64{}
	for i := 0; i < len(arr); i++ {
		y = append(y, math.Pow(y[i]-arr[i], 2))
	}
	total_loss := 0.0
	for _,value:= range y {
		total_loss += value
	}
	return total_loss
}

func SumSquaredRisiduals_derivative(arr []float64, y []float64) []float64 {
	d_loss := []float64{}
	for i := 0; i < len(arr); i++ {
		d_loss = append(d_loss, 2*(arr[i]-y[i]))
	}
	return d_loss
}
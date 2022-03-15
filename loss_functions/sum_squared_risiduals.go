package loss_functions

import "math"

func sum_squared_risiduals(arr []float64, y []float64) float64 {
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
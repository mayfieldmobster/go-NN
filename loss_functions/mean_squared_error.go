package loss_functions

import "math"

func mean_squared_error(arr []float64, y []float64) float64 {
	y = []float64{}
	for i := 0; i < len(arr); i++ {
		y = append(y, math.Pow(arr[i]-y[i], 2))
	}
	total_loss := 0.0
	for _,value:= range y {
		total_loss += value
	}
	mse := total_loss/float64(len(arr))
	return mse
}


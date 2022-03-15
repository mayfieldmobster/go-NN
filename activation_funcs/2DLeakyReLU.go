package activation_funcs


import "math"


func LeakyReLU_2D(arr [][]float64 ) [][]float64 {
	y := [][]float64{}
	for i := 0; i < len(arr); i++ {
		z := []float64{}
		y = append(y, z)
		for j := 0; j < len(arr[i]); j++ {
		y[i] = append(y[i], math.Max(0.1*arr[i][j], arr[i][j]))
		}
	}
	return y
}


package main

import "NN/activation_funcs"
import "fmt"

func main() {
	arr := []float64{1, 2, 3, 4, 5}
	fmt.Println(activation_funcs.Sigmoid(arr))
}
package main

import (
	"fmt"
	"NN/mnist"
)
func printData(dataSet *mnist.DataSet, index int) {
	data := dataSet.Data[index]
	fmt.Println(data.Digit)			// print Digit (label)
	mnist.PrintImage(data.Image)	// print Image
}
func main() {
	dataSet, err := mnist.ReadTrainSet("c:/Users/Chris/go/src/NN/data")
	// or dataSet, err := mnist.ReadTestSet("./mnist")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(dataSet.N)		// number of data
	fmt.Println(dataSet.W)		// image width [pixel]
	fmt.Println(dataSet.H)		// image height [pixel]
	for i := 0; i < 10; i++ {
		printData(dataSet, i)
	}
	printData(dataSet, dataSet.N-1)
	fmt.Println(dataSet.Data[0].Image)
}
package main

import (
	"fmt"
	"NN/mnist"
	"math/rand"
	"time"
)
func printData(dataSet *mnist.DataSet, index int) {
	data := dataSet.Data[index]
	fmt.Println(data.Digit)			// print Digit (label)
	mnist.PrintImage(data.Image)	// print Image
}
func main2() {
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
	//fmt.Println(dataSet.Data[0].Image)
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < 10; i++ {
		fmt.Println(rand.Intn(2))
	}
	num_of_digits := []int64{0,0,0,0,0,0,0,0,0,0}
	for _,data := range dataSet.Data{
		num_of_digits[data.Digit] += 1
	}
	for i,num := range num_of_digits{
		fmt.Println(i,num)
	}
}
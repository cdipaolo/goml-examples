package main

import (
	"fmt"
	"time"

	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/linear"

	mnist "github.com/petar/GoMNIST"
)

// maxI returns the index of the maximum value
// of a slice of float64's
func maxI(array []float64) int {
	var i int
	for j := range array {
		if array[j] > array[i] {
			i = j
		}
	}
	return i
}

func main() {

	fmt.Printf("Loading MNIST dataset into memory...\n")

	train, test, err := mnist.Load("./")
	if err != nil {
		panic(fmt.Sprintf("Error loading training set!\n\t%v\n\n", err))
	}

	fmt.Printf("MNIST dataset loaded!\n\t%v Training Examples\n\t%v Test Examples\n", len(train.Images), len(test.Images))

	tr := train.Sweep()

	stream := make(chan base.Datapoint, 1000)
	errors := make(chan error, 200)

	now := time.Now()
	fmt.Printf("Starting Loading/Training at %v\n", now)

	model := linear.NewSoftmax(base.StochasticGA, 0.1, 0, 10, 0, nil, nil, 784)

	go model.OnlineLearn(errors, stream, func(theta [][]float64) {})

	// push data onto the stream while waiting for errors
	go func() {
		bytes, label, present := tr.Next()
		fmt.Println("\n\n", len(bytes), int(label), []uint8(bytes))
		ct := 1

		for present {
			x := make([]float64, len(bytes))
			y := []float64{float64(label)}

			for i := range bytes {
				x[i] = float64(bytes[i]) / 256
			}

			stream <- base.Datapoint{
				X: x,
				Y: y,
			}

			bytes, label, present = tr.Next()
			ct++

			if ct%500 == 0 {
				print(".")
			}
		}

		fmt.Printf("Loaded %v examples onto the data stream\n", ct)

		close(stream)
	}()

	var more bool
	for {
		err, more = <-errors
		if err != nil {
			fmt.Printf("Error encountered when training!\n\t%v\n", err)
		}
		if !more {
			break
		}
	}

	// now the model is trained! Test it!
	fmt.Printf("Stopped Loading/Training at %v\n\tTraining Time: %v\n", time.Now(), time.Now().Sub(now))

	testEx := test.Count()

	var count int
	var wrong int
	errMap := make(map[int]int32)

	bytes, label := test.Get(0)

	x := make([]float64, len(bytes))

	for i := range bytes {
		x[i] = float64(bytes[i]) / 256
	}

	now = time.Now()
	fmt.Printf("Starting Testing at %v\n", now)

	for ; count < testEx; count++ {
		x := make([]float64, len(bytes))

		for i := range bytes {
			x[i] = float64(bytes[i]) / 256
		}

		guess, err := model.Predict(x)
		if err != nil {
			fmt.Printf("Encountered error while predicting!\n\t%v\n", err)
			continue
		}

		class := maxI(guess)
		if class != int(label) {
			wrong++
			errMap[class]++
		}

		if count%500 == 0 {
			// print(".")
			fmt.Printf("%v percent completed\n", 100*(float64(count)/float64(testEx)))
		}

		bytes, label = test.Get(count)
	}

	fmt.Printf("Stopped Testing at %v\n\tTraining Time: %v\n", time.Now(), time.Now().Sub(now))

	accuracy := 100 * (1 - float64(wrong)/float64(count))
	fmt.Printf("Accuracy: %v percent\n\tPoints Tested: %v\n\tMisclassifications: %v\n\tMisclassifications by Digit: %v\n", accuracy, count, wrong, errMap)
}

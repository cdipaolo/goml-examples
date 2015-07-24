package main

import (
	"fmt"
	"time"

	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/linear"

	"github.com/cheggaaa/pb"
	mnist "github.com/petar/GoMNIST"
)

const (
	// Epochs defines the number of
	// iterations the model will go
	// through the data
	Epochs int64 = 20
)

func newBar(count int64) *pb.ProgressBar {
	bar := pb.New64(count)

	bar.ShowTimeLeft = true
	bar.ShowSpeed = true

	return bar
}

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

func onlineLearn(train *mnist.Set) *linear.Softmax {
	var err error

	stream := make(chan base.Datapoint, 1000)
	errors := make(chan error, 200)

	now := time.Now()
	fmt.Printf("Starting Loading/Training of < Online Softmax > at %v\n", now)

	model := linear.NewSoftmax(base.StochasticGA, 1e-5, 10, 10, 0, nil, nil, 784)

	go model.OnlineLearn(errors, stream, func(theta [][]float64) {})

	// push data onto the stream while waiting for errors
	go func() {
		size := train.Count()
		bytes, label := train.Get(0)

		bar := newBar(int64(size) * Epochs)
		bar.Start()
		count := 0
		for iter := int64(0); iter < Epochs; iter++ {
			for count = 0; count < size; count++ {
				x := make([]float64, len(bytes))
				y := []float64{float64(label)}

				for i := range bytes {
					//x[i] = float64(bytes[i]) / 255
					if bytes[i] > uint8(200) {
						x[i] = 1
					}
				}

				stream <- base.Datapoint{
					X: x,
					Y: y,
				}

				bytes, label = train.Get(count)

				bar.Increment()
			}
		}

		bar.FinishPrint(fmt.Sprintf("Loaded %v examples onto the data stream of < Online Softmax >\n\tRepeated through %v epochs", count, Epochs))

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
	fmt.Printf("Stopped Loading/Training of < Online Softmax > at %v\n\tTraining Time: %v\n", time.Now(), time.Now().Sub(now))

	return model
}

func batchLearn(train *mnist.Set) *linear.Softmax {
	var err error
	x := [][]float64{}
	y := []float64{}

	size := train.Count()
	bytes, label := train.Get(0)

	bar := newBar(int64(size))
	bar.Start()
	count := 0

	now := time.Now()
	fmt.Printf("Starting Loading of < Batch Softmax > at %v\n", now)

	for count = 0; count < size; count++ {
		inp := make([]float64, len(bytes))
		out := float64(label)

		for i := range bytes {
			//x[i] = float64(bytes[i]) / 255
			if bytes[i] > uint8(167) {
				inp[i] = 1
			}
		}

		x = append(x, inp)
		y = append(y, out)

		bytes, label = train.Get(count)

		bar.Increment()
	}

	bar.FinishPrint(fmt.Sprintf("Loaded %v examples into < Batch Softmax >\n\tTook %v", count, time.Now().Sub(now)))

	model := linear.NewSoftmax(base.StochasticGA, 1e-5, 10, 10, int(Epochs), x, y)

	now = time.Now()
	fmt.Printf("Starting Training of < Batch Softmax > at %v over %v epochs\n", now, Epochs)

	err = model.Learn()
	if err != nil {
		panic(fmt.Sprintf("Error learning on < Batch Softmax >!\n\t%v", err))
	}

	fmt.Printf("Stopped Training < Batch Softmax > at %v\n\tTraining Time: %v\n", time.Now(), time.Now().Sub(now))

	return model
}

func eval(model *linear.Softmax, test *mnist.Set, method string) {
	testEx := test.Count()

	var count int
	var wrong int
	errMap := make(map[int]int32)

	bytes, label := test.Get(0)

	now := time.Now()
	fmt.Printf("Starting Testing of < %v > at %v\n", method, now)

	bar := newBar(int64(testEx))
	bar.Start()

	for ; count < testEx; count++ {
		x := make([]float64, len(bytes))

		for i := range bytes {
			//x[i] = float64(bytes[i]) / 255
			if bytes[i] > uint8(220) {
				x[i] = 1
			}
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

		bar.Increment()
		bytes, label = test.Get(count)
	}

	bar.FinishPrint(fmt.Sprintf("Stopped Testing of < %v > at %v\n\tTraining Time: %v\n", method, time.Now(), time.Now().Sub(now)))

	accuracy := 100 * (1 - float64(wrong)/float64(count))
	fmt.Printf("Accuracy of < %v >: %v percent\n\tPoints Tested: %v\n\tMisclassifications: %v\n\tMisclassifications by Digit: %v\n", method, accuracy, count, wrong, errMap)
}

func main() {

	fmt.Printf("Loading MNIST dataset into memory...\n")

	train, test, err := mnist.Load("./")
	if err != nil {
		panic(fmt.Sprintf("Error loading training set!\n\t%v\n\n", err))
	}

	fmt.Printf("MNIST dataset loaded!\n\t%v Training Examples\n\t%v Test Examples\n", len(train.Images), len(test.Images))

	//online := onlineLearn(train)

	//eval(online, test, "Online Softmax")

	batch := batchLearn(train)

	eval(batch, test, "Batch Softmax")
}

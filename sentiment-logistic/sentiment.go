package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"golang.org/x/text/transform"
)

var (
	// words is the main dictionary
	// we're using to base our features
	// from
	words map[string]int
	count int

	// filepaths
	amz string
	ylp string
	imd string

	sanitize transform.Transformer
)

func init() {
	words = make(map[string]int)
	count = 0

	amz = "./amazon_cells_labelled.txt"
	ylp = "./yelp_labelled.txt"
	imd = "./imdb_labelled.txt"

	sanitize = transform.RemoveFunc(func(r rune) bool {
		switch {
		case r >= 'A' && r <= 'Z':
			return false
		case r >= 'a' && r <= 'z':
			return false
		case r >= '0' && r <= '1':
			return false
		case r == ' ':
			return false
		case r == '\t':
			return false
		default:
			return true
		}
	})

	rand.Seed(42)
}

func parseLinesToData(filepath string) ([][]float64, []float64) {
	f, err := os.Open(filepath)
	if err != nil {
		panic(err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanLines)

	x := [][]float64{}
	y := []float64{}

	now := time.Now()
	fmt.Printf("Starting munging from < %v > to data at %v\n", filepath, now)

	var ct int

	for scanner.Scan() {
		// remove all punctuation and convert to lower case
		scanText := scanner.Text()

		line := parseLineToText(scanText)

		text := strings.Split(line[0], " ")

		ct++
		if ct%500 == 0 {
			print(".")
		}

		label, err := strconv.Atoi(line[1])
		if err != nil {
			continue
		}

		l := make([]float64, len(words))

		for _, txt := range text {

			// skip small words
			if len(l) < 4 {
				continue
			}

			l[words[txt]] = 1.0
		}

		x = append(x, l)
		y = append(y, float64(label))

	}
	fmt.Printf("\nFinished munging from < %v > to data\n\tdelta: %v\n", filepath, time.Now().Sub(now))

	return x, y
}

// addWordsToGlobalMap adds words contained within
// the dataset's expected format to the global
// `words` dictionary variable
func addWordsToGlobalMap(filepath string) {
	f, err := os.Open(filepath)
	if err != nil {
		panic(err.Error())
	}
	defer f.Close()

	// create word bank
	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanLines)

	now := time.Now()
	fmt.Printf("Starting adding words from < %v > to word bank at %v\n", filepath, now)

	for scanner.Scan() {
		// remove all punctuation and convert to lower case
		text := strings.Split(parseLineToText(scanner.Text())[0], " ")

		for i := range text {

			// skip small words
			if len(text[i]) < 4 {
				continue
			}

			// add word to map if the word isn't present
			if _, there := words[text[i]]; !there {
				words[text[i]] = count
				count++
			}
		}
	}
	fmt.Printf("Finished adding words from < %v > to bank\n\tdelta: %v\n", filepath, time.Now().Sub(now))
}

// parseLineToText takes in a string, converts
// it to lowercase, removes anything not in a-z,
// and returns that, split between the sentence
// and the label
func parseLineToText(s string) []string {
	sanitized, _, _ := transform.String(sanitize, s)

	return strings.Split(strings.TrimSuffix(strings.ToLower(sanitized), "\n"), "\t")
}

func parseDataIntoPoints() ([][]float64, []float64) {
	// add words from files to dictionary
	addWordsToGlobalMap(amz)
	addWordsToGlobalMap(ylp)
	addWordsToGlobalMap(imd)

	// now go through and add each sentence to the array
	x := [][]float64{}
	y := []float64{}

	az, azY := parseLinesToData(amz)
	x = append(x, az...)
	y = append(y, azY...)

	yl, ylY := parseLinesToData(ylp)
	x = append(x, yl...)
	y = append(y, ylY...)

	im, imY := parseLinesToData(imd)
	x = append(x, im...)
	y = append(y, imY...)

	// shuffle data to mix imdb, amazon, and yelp reviews together

	now := time.Now()
	fmt.Printf("Shuffling the data now!\n\tStarted: %v\n", now)

	for i := range x {
		j := rand.Intn(i + 1)
		x[i], x[j] = x[j], x[i]
	}

	for i := range y {
		j := rand.Intn(i + 1)
		y[i], y[j] = y[j], y[i]
	}

	fmt.Printf("Finshed shuffling!\n\tTook %v\n", time.Now().Sub(now))

	return x, y
}

func main() {
	now := time.Now()
	fmt.Printf("Parsing Data Into Datapoints\n\tStarting at %v\n", now)

	x, y := parseDataIntoPoints()

	fmt.Printf("Parsing Data Finished!\n\tEnding in %v\n\t%v, %v datapoints recorded for x,y\n", time.Now().Sub(now), len(x), len(y))
}

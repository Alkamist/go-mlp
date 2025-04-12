package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	_ "image/png"
	"io"
	"log"
	"net/http"
	"os"
)

const serverURL = "http://localhost:30080/predict"

type InputData struct {
	Values []float32 `json:"values"`
}

type ResultData struct {
	Result int `json:"result"`
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run client.go <path-to-image.png>")
		os.Exit(1)
	}

	imagePath := os.Args[1]
	
	// Read and process the image
	floatValues, err := processImage(imagePath)
	if err != nil {
		log.Fatalf("Failed to process image: %v", err)
	}

	// Send to server and get result
	result, err := sendToServer(floatValues)
	if err != nil {
		log.Fatalf("Failed to get prediction: %v", err)
	}

	fmt.Printf("Prediction result: %d\n", result)
}

// processImage reads a PNG image and converts it to an array of 784 floats
func processImage(filePath string) ([]float32, error) {
	// Open the image file
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open image file: %w", err)
	}
	defer file.Close()

	// Decode the image
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	// Ensure image is 28x28
	bounds := img.Bounds()
	if bounds.Dx() != 28 || bounds.Dy() != 28 {
		return nil, fmt.Errorf("image must be 28x28 pixels, got %dx%d", bounds.Dx(), bounds.Dy())
	}

	// Convert image to array of 784 float values (0.0-1.0)
	values := make([]float32, 784)
	idx := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			pixel := img.At(x, y)
			r, _, _, _ := color.GrayModel.Convert(pixel).RGBA()

			grayValue := float32(r) / 65535.0
			
			values[idx] = grayValue
			idx++
		}
	}

	return values, nil
}

// sendToServer sends the float values to the server and returns the prediction
func sendToServer(values []float32) (int, error) {
	// Create request payload
	inputData := InputData{Values: values}
	jsonData, err := json.Marshal(inputData)
	if err != nil {
		return 0, fmt.Errorf("failed to encode input data: %w", err)
	}

	// Send POST request to server
	resp, err := http.Post(serverURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return 0, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return 0, fmt.Errorf("server returned error status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	// Parse response
	var resultData ResultData
	if err := json.NewDecoder(resp.Body).Decode(&resultData); err != nil {
		return 0, fmt.Errorf("failed to decode response: %w", err)
	}

	return resultData.Result, nil
}
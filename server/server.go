package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	ml "rancher-test/machine_learning"
)

const expectedInputSize = 784

var model *ml.Mlp

type InputData struct {
	Values []float32 `json:"values"`
}

type ResultData struct {
	Result int `json:"result"`
}

func processInput(values []float32) int {
	output := model.Forward(values)
	return ml.Argmax(output[:])
}

func handlePrediction(w http.ResponseWriter, r *http.Request) {
	// Only accept POST requests
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse the request body
	var input InputData
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate input size
	if len(input.Values) != expectedInputSize {
		http.Error(w, fmt.Sprintf("Expected %d float values, got %d", expectedInputSize, len(input.Values)), http.StatusBadRequest)
		return
	}

	result := processInput(input.Values)

	// Prepare and send the response
	response := ResultData{Result: result}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Error encoding response: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}
}

func main() {
	model = &ml.Mlp{}
	model.Init(expectedInputSize, 100, 10)
	model.Load("models/model.json")

	http.HandleFunc("/predict", handlePrediction)

	port := ":8080"
	fmt.Printf("Server listening on port %s...\n", port)
	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}
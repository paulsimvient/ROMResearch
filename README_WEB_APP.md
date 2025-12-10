# ROM Model Tester Web App

A simple web interface for testing Reduced Order Models (ROMs).

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

```bash
source venv/bin/activate
python3 web_app.py
```

The app will start on http://localhost:5000

## Usage

1. View available ROM types in the first section
2. Generate training data by specifying features and samples
3. Create a ROM model:
   - Select ROM type (DMD, Koopman, NeuralNetwork, Autoencoder)
   - Enter a unique model ID
   - Optionally provide parameters as JSON (e.g., `{"rank": 10, "implementation": "numpy"}`)
4. Fit the model with the generated training data
5. Run simulations:
   - Select a fitted model
   - Provide initial condition (comma-separated values or JSON array)
   - Specify number of steps
   - View results in the plot

## Example Workflow

1. Generate data: 50 features, 200 samples
2. Create DMD ROM: type "DMD", ID "test_dmd", params `{"rank": 10}`
3. Fit the model with generated data
4. Simulate: initial condition `0.5, 0.3, 0.1, ...` (or use first few values from training data), 100 steps

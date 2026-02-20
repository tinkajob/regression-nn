# Price Prediction AI

A simple neural network project which allows you to train  models to predict house prices using evolutionary methods.

## Features
- Predicts house prices based on features like bedrooms, bathrooms, sqft, etc.
- Targets are log-scaled for stable training
- Evolutionary algorithm with adjustable mutation strength
- Optional ReLU activation in hidden layers
- Evaluates predictions in real dollars

## Usage
1. Prepare a CSV dataset with columns:
price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement, yr_built, yr_renovated
2. Define training parameters (JSON file)
3. Run the training script.
4. Check the Mean Absolute Error (MAE) in dollars after training is done.
5. Load your model and predict prices.

## Notes
- Features are normalized; target prices are log-scaled (or normalized)
- Output layer is linear; hidden layers can use Leaky ReLU
- Mutation strength can gradually decrease over generations for smoother convergence

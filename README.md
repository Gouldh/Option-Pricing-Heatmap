# Option Pricing Models with Visualization

## Project Overview
This repository offers a financial analysis tool that computes option prices using sophisticated models like Black-Scholes, Heston, and Merton Jump Diffusion, and provides a heatmap showing the effects of different volatilities and interest rates on option price. A version of this project [written in C++](https://github.com/Gouldh/Option-Pricing-CPP) also exists, however it does not offer the heatmap functionality.

## Features
- **Black-Scholes Model**: Applies the Black-Scholes formula for option pricing.
- **Heston Model**: Includes stochastic volatility in the pricing model to capture intricate market dynamics.
- **Merton Jump Diffusion Model**: Considers abrupt price changes in asset valuation for more accurate option pricing.
- **Heatmap Visualization**: Creates heatmaps to illustrate option price variations across different volatilities and interest rates, providing clear visual data interpretation.

## Libraries
- `numpy`: Performs computation of mathematical functions. Used in every model for calculating option price.
- `scipy`: Used for its statistical functions. Used to calculate the CDF of the Normal Distribution in the Black-Sholes model.
- `matplotlib`: Required for data visualization. Used for generating heatmap with varying options prices for varying parameters.

## Installation
Follow these steps to set up the tool:

1. Clone the repository:
   ```bash
   git clone https://github.com/Gouldh/Option-Pricing-Heatmap.git
   ```
2. Navigate to the repository's directory:
   ```bash
   cd Option-Pricing-Heatmap
   ```
3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
To run the pricing models and generate visualizations, execute the `main.py` script:

```bash
python main.py
```

## Sample Output
Below is an example of the output produced by running the code with sample input parameters. The text represents generated values using the example parameters. The Heatmap shows the options price for varying volatilities and interest rates.

```plaintext
Black-Scholes Price: $7.26
Heston Model Price: $6.37
Merton Jump Diffusion Price: $6.50
```

![Example output](https://github.com/Gouldh/Option-Pricing-Heatmap/blob/main/Option%20Pricing%20Heatmap%20Example%20Output.png)

## License
This project is open-sourced under the MIT License. For more information, refer to the `LICENSE` file.

**Author**: Hunter Gould         
**Date**: 11/24/2023

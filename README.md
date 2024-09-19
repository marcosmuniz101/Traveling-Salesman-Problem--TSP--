# Metaheuristic_L1
 
# Traveling Salesman Problem (TSP) Solution Generator

## Project Overview

This project implements heuristic solutions for the Traveling Salesman Problem (TSP). It includes three algorithms to find a near-optimal tour for a given set of cities:

1. **Nearest Neighbor Insertion**: Starts from a random city and iteratively inserts the closest unvisited city into the tour.
2. **Alternative Insertion Heuristic**: Starts from a random city, chooses a random unvisited city, and inserts it after the closest city in the current tour.
3. **Random Tour**: Generates a random permutation of all cities.

The project also generates visualizations of the tours and compiles results into an HTML report.

## Features

- Implemented Nearest Neighbor and Alternative Insertion heuristics.
- Random tour generation.
- Visualization of tours using `matplotlib`.
- HTML report generation with performance metrics and tour visualizations.

## Prerequisites

Ensure you have Python 3.x installed along with the following Python libraries:

- `numpy`
- `matplotlib`
- `psutil`

You can install the required libraries using pip:

```sh
pip install numpy matplotlib psutil
```

## Usage

1. **Prepare the Input File**: Create a text file with your city data in the following format:
   
n Id_0 x_0 y_0 Id_1 x_1 y_1 ... Id_n-1 x_n-1 y_n-1

- `n` is the number of cities.
- Each subsequent line contains the city ID and its x and y coordinates.

2. **Run the Script**: Execute the script using the command line. Provide the input filename and the number of runs as arguments:

```sh
python tsp.py <input_filename> <nRuns>
```

Por exemplo: 
```sh
python tsp.py inst-1.tsp 100

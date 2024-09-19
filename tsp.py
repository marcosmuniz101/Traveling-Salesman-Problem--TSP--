import math
import os
import sys
import random
import time
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
import psutil


#Read the file with cities and their coordinates
def read_tsp(file):
    with open(file, 'r') as f:
        n = int(f.readline().strip())  # Read the number of cities
        list_cities = []

        # Read the next n lines with city coordinates
        for _ in range(n):
            line = f.readline().strip().split()
            # Parse x and y coordinates (ignore the city ID)
            x, y = float(line[1]), float(line[2])
            list_cities.append([x, y])
    #print (np.array(list_cities))
    return np.array(list_cities)  # Return as a numpy array for easy distance calculations

# Function to calculate Euclidean distance between two points
def distance(city1, city2):
    # Euclidean distance formula with rounding to nearest integer
    #print (round(math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)))
    return round(math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2))

# Function to calculate the total tour distance
def total_distance(tour, cities):
    total = 0
    for i in range(len(tour) - 1):
        total += distance(cities[tour[i]], cities[tour[i+1]])  # Distance between consecutive cities
    total += distance(cities[tour[-1]], cities[tour[0]])  # Distance to return to the starting city (cycle)
    return total

# Function that implements the NN in the list of cities
def nearest_neighbor(cities):
    n = len(cities)
    unvisited = list(range(n))
    start_city = random.choice(unvisited)
    tour = [start_city]
    unvisited.remove(start_city)

    while unvisited:
        current_city = tour[-1]
        nearest_city = min(unvisited, key=lambda city: distance(cities[current_city], cities[city]))
        tour.append(nearest_city)
        unvisited.remove(nearest_city)

    return tour


def alternative(cities):
    n = len(cities)

    # Start with a random city as the initial tour
    initial_city = random.randint(0, n - 1)
    tour = [initial_city]
    unvisited = set(range(n)) - {initial_city}  # Set of unvisited cities

    while unvisited:
        # Choose a random unvisited city
        random_city = random.choice(list(unvisited))

        # Find the city in the current tour that is closest to the random city
        min_distance = float('inf')
        insert_position = 0

        for i in range(len(tour)):
            city_in_tour = tour[i]
            dist = distance(cities[random_city], cities[city_in_tour])
            if dist < min_distance:
                min_distance = dist
                insert_position = i

        # Insert the random city after the closest city in the tour
        tour.insert(insert_position + 1, random_city)

        # Mark the random city as visited
        unvisited.remove(random_city)

    return tour

def random_tour(cities):
    tour = list(range(len(cities)))
    random.shuffle(tour)
    return tour

def format_dist(distance):
    # Format the distance with thousands separators and one decimal place
    formatted_distance = f"{(distance / 1000):,.2f}"
    return f"{formatted_distance} km"

def plot_tour(cities, tour, name):
    # Extract x and y coordinates of the cities
    x_coords = cities[:, 0]
    y_coords = cities[:, 1]

    # Get the coordinates for the tour
    tour_x = [x_coords[city] for city in tour] + [x_coords[tour[0]]]  # Close the loop
    tour_y = [y_coords[city] for city in tour] + [y_coords[tour[0]]]  # Close the loop

    plt.figure(figsize=(13, 10))
    plt.plot(tour_x, tour_y, marker='o', linestyle='-', color='b')  # Plot the tour
    plt.scatter(x_coords, y_coords, color='r')  # Plot the cities

    # Highlight the starting city
    start_city = tour[0]
    plt.scatter(x_coords[start_city], y_coords[start_city], color='g', s=100, edgecolor='black',
                zorder=5)  # Larger size and different color

    plt.title('TSP Tour Visualization by ' + name, fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)

    # Annotate city points
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.text(x, y, str(i + 1), fontsize=12, ha='right')

        # Show direction with arrows
    for i in range(len(tour) - 1):
        start_city = tour[i]
        end_city = tour[i + 1]
        plt.annotate('', xy=(x_coords[end_city], y_coords[end_city]),
                     xytext=(x_coords[start_city], y_coords[start_city]),
                     arrowprops=dict(facecolor='blue', edgecolor='black', arrowstyle='->', linewidth=2))
    plt.grid(True)

    # Save the plot as an image
    output_dir = 'tsp_best_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = f'{output_dir}/{name}.png'
    plt.savefig(file_path)
    plt.close()


def generate_html_with_best_plots(infos, n_runs, n_cities):
    html_content = '''<html>
    <head>
        <title>Best TSP Tours</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                color: #333;
                margin: 20px;
            }
            h1 {
                text-align: center;
                font-size: 36px;
                color: #4CAF50;
            }
            h2 {
                color: #333;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 10px;
            }
            h3, h4 {
                margin: 10px 0;
                font-weight: normal;
            }
            .container {
                width: 80%;
                margin: auto;
                background-color: #fff;
                padding: 20px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            }
            .tour-section {
                margin-bottom: 40px;
                border: 1px solid #ddd;
                padding: 20px;
                border-radius: 10px;
                background-color: #fafafa;
            }
            img {
                width: 100%;
                height: auto;
                border-radius: 10px;
                margin: 20px 0;
            }
            .footer {
                text-align: center;
                margin-top: 20px;
                color: #777;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Best Generated TSP Tours by Algorithm</h1>
            <h3>Each Algorithm ran {n_runs} times with a list of {n_cities} cities.</h3>
    '''

    for i in range(len(infos)):
        html_content += f'''
            <div class="tour-section">
                <h2>Algorithm: {infos[i][0]}</h2>
                <h4><strong>Best Distance:</strong> {infos[i][1]},  
                    <strong>Average Distance:</strong> {infos[i][2]},  
                    <strong>Execution Time:</strong> {infos[i][3]:.3f} ms,  
                    <strong>Memory Usage:</strong> {infos[i][4]} bytes
                </h4>
                <img src="tsp_best_plots/{infos[i][0]}.png" alt="Best Tour - {infos[i][0]}">
            </div>
        '''

    html_content += '''
            <div class="footer">
                <p>Generated by TSP Solver - Marcos Oliveira - {time.ctime()}</p>
            </div>
        </div>
    </body>
    </html>
    '''

    # Save and open the HTML file
    html_filename = 'infos.html'
    browser = webbrowser.get()
    with open(html_filename, 'w') as file:
        file.write(html_content)
    browser.open_new_tab('infos.html')





def run_algorithm(cities, n_runs, heuristic):
    best_tour = None
    best_distance = float('inf')
    total_distance_sum = 0

    #time start
    start_time = time.time()
    #memory start
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    # run the algorithm
    for _ in range(n_runs):
        tour = heuristic(cities)
        tour_distance = total_distance(tour, cities)

        if tour_distance < best_distance:
            best_distance = tour_distance
            best_tour = tour

        total_distance_sum += tour_distance

    # time end
    end_time = time.time()
    execution_time = end_time - start_time
    # memory end
    memory_after = process.memory_info().rss
    memory_usage = memory_after - memory_before

    average_distance = total_distance_sum / n_runs

    return best_tour, best_distance, average_distance, execution_time, memory_usage

if __name__ == "__main__":
    filename = sys.argv[1]
    n_runs = int(sys.argv[2])

    cities = read_tsp(filename)

    infos_extra = []


    # Run Nearest Neighbor
    print("------------- Nearest Neighbor TOUR -----------------------------------")
    best_tour, best_distance, avg_distance, execution_time, memory_usage = run_algorithm(cities, n_runs, nearest_neighbor)
    print(f"Best Distance (Nearest Neighbor): {format_dist(best_distance)}")
    print(f"Average Distance (Nearest Neighbor): {format_dist(avg_distance)}")
    print(f"Execution time (Nearest Neighbor): {execution_time * 1000:.3f} ms")
    print(f"Memory usage (Nearest Neightbor): {memory_usage / 1024:.2f} KB")
    print(f"Best Tour (Nearest Neighbor): {best_tour}")
    infos_extra.append(['Nearest Neighbor', format_dist(best_distance), format_dist(avg_distance), execution_time, memory_usage])
    plot_tour(cities, best_tour, 'Nearest Neighbor')

    print ( "------------- Alternative TOUR -----------------------------------")
    best_tour, best_distance, avg_distance, execution_time, memory_usage = run_algorithm(cities, n_runs, alternative)
    print(f"Best Distance (Alternative Tour): {format_dist(best_distance)}")
    print(f"Average Distance (Alternative Tour): {format_dist(avg_distance)}")
    print(f"Execution time (Alternative Tour): {execution_time * 1000:.3f} ms")
    print(f"Memory usage (Alternative Tour): {memory_usage / 1024:.2f} KB")
    print(f"Best Tour (Random): {best_tour}")
    infos_extra.append(['Alternative Tour', format_dist(best_distance), format_dist(avg_distance), execution_time, memory_usage])
    plot_tour(cities, best_tour, 'Alternative Tour')

    print ( "------------- RAMDOM TOUR -----------------------------------")
    best_tour, best_distance, avg_distance, execution_time, memory_usage = run_algorithm(cities, n_runs, random_tour)
    print(f"Best Distance (Random): {format_dist(best_distance)}")
    print(f"Average Distance (Random): {format_dist(avg_distance)}")
    print(f"Execution time (Random Tour): {execution_time * 1000:.3f} ms")
    print(f"Memory usage (Random Tour): {memory_usage / 1024:.2f} KB")
    print(f"Best Tour (Random): {best_tour}")
    infos_extra.append(['Random', format_dist(best_distance), format_dist(avg_distance), execution_time, memory_usage])
    plot_tour(cities, best_tour, 'Random')

    generate_html_with_best_plots(infos_extra, n_runs, len(cities))













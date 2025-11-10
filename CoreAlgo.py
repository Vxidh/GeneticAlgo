import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PLOT_WIDTH = 20
PLOT_HEIGHT = 20
PLOT_AREA = PLOT_WIDTH * PLOT_HEIGHT

ROOMS_SPEC = [
    {'name': 'Living', 'min_w': 5, 'min_h': 5},
    {'name': 'Kitchen', 'min_w': 3, 'min_h': 3},
    {'name': 'Bedroom 1', 'min_w': 4, 'min_h': 4},
    {'name': 'Bathroom', 'min_w': 2, 'min_h': 3},
    {'name': 'Bedroom 2', 'min_w': 4, 'min_h': 4},
]
NUM_ROOMS = len(ROOMS_SPEC)

ADJACENCY_RULES = [
    {'room1': 'Kitchen', 'room2': 'Living', 'desired_dist': 0.0, 'weight': 2.0},
    {'room1': 'Bathroom', 'room2': 'Bedroom 1', 'desired_dist': 0.0, 'weight': 1.0},
    {'room1': 'Bathroom', 'room2': 'Bedroom 2', 'desired_dist': 0.0, 'weight': 1.0},
]
ROOM_NAME_MAP = {spec['name']: i for i, spec in enumerate(ROOMS_SPEC)}


POPULATION_SIZE = 200
GENERATIONS = 200
ELITISM_COUNT = 20
MUTATION_RATE = 0.6
GENE_MUTATION_RATE = 0.3

class Floorplan:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = 0.0

def create_individual():
    """
    Creates a single random Floorplan (chromosome).
    This is your "initial population" generator.
    """
    chromosome = []
    for room_spec in ROOMS_SPEC:
        # Generate random dimensions, ensuring they meet minimums
        w = random.uniform(room_spec['min_w'], room_spec['min_w'] * 2.5) 
        h = random.uniform(room_spec['min_h'], room_spec['min_h'] * 2.5)

        # Generate random coordinates
        x = random.uniform(0, PLOT_WIDTH - w)
        y = random.uniform(0, PLOT_HEIGHT - h)

        chromosome.append({'x': x, 'y': y, 'w': w, 'h': h})
    
    return Floorplan(chromosome)

def create_initial_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

def calculate_overlap(room1, room2):
    dx = min(room1['x'] + room1['w'], room2['x'] + room2['w']) - max(room1['x'], room2['x'])
    dy = min(room1['y'] + room1['h'], room2['y'] + room2['h']) - max(room1['y'], room2['y'])

    if dx > 0 and dy > 0:
        return dx * dy
    return 0

def calculate_fitness(floorplan):
    total_penalty = 0.0
    total_room_area = 0.0

    W_MIN_SIZE = 20.0
    W_OVERLAP = 150.0
    W_BOUNDARY = 10.0
    
    # "Suggestions" - Soft Constraints.
    # We are making these much stronger now.
    W_ADJACENCY = 1.0      # INCREASED: Make adjacency *very* important
    W_ASPECT_RATIO = 0.5   # INCREASED: Make shape *very* important
    W_WHITESPACE = 0.1    # DECREASED: Don't ossess over whitespace


    # --- a) Boundary Penalty Calculation ---
    for room in floorplan.chromosome:
        r_right = room['x'] + room['w']
        r_top = room['y'] + room['h']

        if r_right > PLOT_WIDTH:
            total_penalty += (r_right - PLOT_WIDTH) * room['h'] * W_BOUNDARY
        if r_top > PLOT_HEIGHT:
            total_penalty += (r_top - PLOT_HEIGHT) * room['w'] * W_BOUNDARY
        if room['x'] < 0:
            total_penalty += abs(room['x']) * room['h'] * W_BOUNDARY
        if room['y'] < 0:
            total_penalty += abs(room['y']) * room['w'] * W_BOUNDARY
        
        total_room_area += room['w'] * room['h']

    total_overlap_area = 0
    for i in range(NUM_ROOMS):
        for j in range(i + 1, NUM_ROOMS):
            room1 = floorplan.chromosome[i]
            room2 = floorplan.chromosome[j]
            overlap_area = calculate_overlap(room1, room2)
            if overlap_area > 0:
                total_overlap_area += overlap_area
    
    total_penalty += total_overlap_area * W_OVERLAP

    for i, room in enumerate(floorplan.chromosome):
        spec = ROOMS_SPEC[i]
        if room['w'] < spec['min_w']:
            total_penalty += (spec['min_w'] - room['w']) * W_MIN_SIZE
        if room['h'] < spec['min_h']:
            total_penalty += (spec['min_h'] - room['h']) * W_MIN_SIZE

    whitespace_area = PLOT_AREA - total_room_area + total_overlap_area
    if whitespace_area > 0:
        total_penalty += whitespace_area * W_WHITESPACE
    
    max_aspect_ratio = 3.0
    for room in floorplan.chromosome:
        if room['w'] < 1e-6 or room['h'] < 1e-6: continue
        
        ratio = max(room['w'] / room['h'], room['h'] / room['w'])
        if ratio > max_aspect_ratio:
            total_penalty += (ratio - max_aspect_ratio) * W_ASPECT_RATIO
    
    for rule in ADJACENCY_RULES:
        idx1 = ROOM_NAME_MAP[rule['room1']]
        idx2 = ROOM_NAME_MAP[rule['room2']]
        room1 = floorplan.chromosome[idx1]
        room2 = floorplan.chromosome[idx2]

        center1_x = room1['x'] + room1['w'] / 2
        center1_y = room1['y'] + room1['h'] / 2
        center2_x = room2['x'] + room2['w'] / 2
        center2_y = room2['y'] + room2['h'] / 2
        
        # --- This is the new "edge-to-edge" logic ---
        # Calculate the gap between the boxes on x and y axes
        dx = max(0, abs(center1_x - center2_x) - (room1['w'] / 2 + room2['w'] / 2))
        dy = max(0, abs(center1_y - center2_y) - (room1['h'] / 2 + room2['h'] / 2))
        
        dist = math.sqrt(dx**2 + dy**2)

        penalty = abs(dist - rule['desired_dist']) * rule['weight'] * W_ADJACENCY
        if penalty > 0:
            total_penalty += penalty

    floorplan.fitness = 1.0 / (1.0 + total_penalty)


def selection(population):
    def tournament():
        tournament_contenders = random.sample(population, 5)
        return max(tournament_contenders, key=lambda ind: ind.fitness)

    parent1 = tournament()
    parent2 = tournament()
    return parent1, parent2

def crossover(parent1, parent2):
    child_chromosome = []
    for i in range(NUM_ROOMS):
        if random.random() < 0.5:
            child_chromosome.append(parent1.chromosome[i].copy())
        else:
            child_chromosome.append(parent2.chromosome[i].copy())
            
    return Floorplan(child_chromosome)

def mutate(floorplan):
    """
    Performs 'Creep Mutation' on the floorplan.
    """
    for i in range(NUM_ROOMS):
        if random.random() < MUTATION_RATE:
            room = floorplan.chromosome[i]
            spec = ROOMS_SPEC[i]

            if random.random() < GENE_MUTATION_RATE:
                room['x'] += random.uniform(-1, 1) * (PLOT_WIDTH * 0.1)
                room['x'] = max(0, min(room['x'], PLOT_WIDTH - room['w']))
            if random.random() < GENE_MUTATION_RATE:
                room['y'] += random.uniform(-1, 1) * (PLOT_HEIGHT * 0.1)
                room['y'] = max(0, min(room['y'], PLOT_HEIGHT - room['h']))
            if random.random() < GENE_MUTATION_RATE:
                room['w'] += random.uniform(-1, 1) * (spec['min_w'] * 0.5)
                room['w'] = max(spec['min_w'], room['w'])
                room['x'] = max(0, min(room['x'], PLOT_WIDTH - room['w']))
            if random.random() < GENE_MUTATION_RATE:
                room['h'] += random.uniform(-1, 1) * (spec['min_h'] * 0.5)
                room['h'] = max(spec['min_h'], room['h'])
                room['y'] = max(0, min(room['y'], PLOT_HEIGHT - room['h']))
    
    return floorplan

def visualize_floorplan(floorplan, generation, title_prefix=""):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, PLOT_WIDTH)
    ax.set_ylim(0, PLOT_HEIGHT)
    ax.set_aspect('equal')

    boundary = patches.Rectangle((0, 0), PLOT_WIDTH, PLOT_HEIGHT, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(boundary)

    cmap = plt.colormaps['tab10']
    colors = cmap(range(NUM_ROOMS))

    for i, room in enumerate(floorplan.chromosome):
        spec = ROOMS_SPEC[i]
        rect = patches.Rectangle(
            (room['x'], room['y']), 
            room['w'], 
            room['h'], 
            linewidth=1, 
            edgecolor='black', 
            facecolor=colors[i], 
            alpha=0.7
        )
        ax.add_patch(rect)
        ax.text(
            room['x'] + room['w'] / 2, 
            room['y'] + room['h'] / 2, 
            f"{spec['name']}\n({room['w']:.1f}x{room['h']:.1f})", 
            ha='center', 
            va='center', 
            fontsize=8
        )
    
    plt.title(f"{title_prefix}Generation {generation} - Fitness: {floorplan.fitness:.4f}")
    plt.savefig(f"{title_prefix.lower()}generation_{generation}.png")
    plt.close(fig)


def main():
    print("Starting Genetic Algorithm for Floorplan Optimization...")
    
    population = create_initial_population()

    for gen in range(GENERATIONS):
        for individual in population:
            calculate_fitness(individual)
        
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        best_individual = population[0]
        
        print(f"Gen {gen:03d}: Best Fitness = {best_individual.fitness:.6f}")
        
        if gen % (GENERATIONS // 10) == 0 or gen == GENERATIONS - 1:
            visualize_floorplan(best_individual, gen)

        new_population = []
        
        new_population.extend(population[:ELITISM_COUNT])
        
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = selection(population)
            child = crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)
        
        population = new_population

    print("...Optimization Finished.")
    
    final_best = population[0]
    calculate_fitness(final_best)
    print(f"Final Best Fitness: {final_best.fitness}")
    visualize_floorplan(final_best, GENERATIONS, title_prefix="FINAL_")
    print(f"Final visualization saved to 'FINAL_generation_{GENERATIONS}.png'")

if __name__ == "__main__":
    main()
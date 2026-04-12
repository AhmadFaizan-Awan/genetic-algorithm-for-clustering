import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import StandardScaler

class GeneticAlgorithmClustering:
    def __init__(self, n_clusters, n_population=50, max_generations=100, mutation_rate=0.1, selection_rate=0.5):
        self.k = n_clusters
        self.pop_size = n_population
        self.max_gen = max_generations
        self.mutation_rate = mutation_rate
        self.selection_rate = selection_rate
        self.population = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []

    def initialize_population(self, data):
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        self.population = []
        for _ in range(self.pop_size):
            centroids = []
            for _ in range(self.k):
                 centroid = np.random.uniform(min_vals, max_vals, size=data.shape[1])
                 centroids.append(centroid)
            self.population.append(np.array(centroids))

    def assign_clusters(self, data, centroids):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def calculate_fitness(self, data, centroids):
        labels = self.assign_clusters(data, centroids)
        fitness = 0
        for i in range(self.k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                fitness += np.sum(np.linalg.norm(cluster_points - centroids[i], axis=1)**2)
        return fitness

    def selection(self, data, fitness_scores):
        selected_indices = []
        num_parents = int(self.pop_size * self.selection_rate)
        for _ in range(num_parents):
            tournament = np.random.choice(len(self.population), size=3, replace=False)
            best_idx = tournament[np.argmin([fitness_scores[i] for i in tournament])]
            selected_indices.append(best_idx)
        return [self.population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        split = np.random.randint(1, self.k)
        child1 = np.concatenate((parent1[:split], parent2[split:]))
        child2 = np.concatenate((parent2[:split], parent1[split:]))
        return child1, child2

    def mutate(self, chromosome, data):
        if np.random.rand() < self.mutation_rate:
            idx = np.random.randint(0, self.k)
            noise = np.random.normal(0, 0.1, size=chromosome[idx].shape)
            chromosome[idx] += noise
        return chromosome

    def fit(self, data, progress_callback=None):
        self.initialize_population(data)
        
        for gen in range(self.max_gen):
            fitness_scores = [self.calculate_fitness(data, ind) for ind in self.population]
            
            min_fitness = min(fitness_scores)
            best_idx = np.argmin(fitness_scores)
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                self.best_solution = self.population[best_idx]
            self.history.append(self.best_fitness)
            
            parents = self.selection(data, fitness_scores)
            
            new_population = []
            new_population.append(self.best_solution)
            
            while len(new_population) < self.pop_size:
                p1, p2 = np.random.choice(len(parents), 2, replace=False)
                child1, child2 = self.crossover(parents[p1], parents[p2])
                new_population.append(self.mutate(child1, data))
                if len(new_population) < self.pop_size:
                     new_population.append(self.mutate(child2, data))
            
            self.population = new_population
            
            if progress_callback:
                progress_callback(gen + 1, self.max_gen, self.best_fitness)
                
        return self.best_solution, self.history

def load_iris_data():
    data = load_iris()
    X = data.data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, "Iris Dataset"

def load_blobs_data(n_samples=300, centers=4, random_state=42):
    X, _ = make_blobs(n_samples=n_samples, centers=centers, n_features=2, random_state=random_state)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, "Synthetic Blobs"

def load_mall_data(file_path="Mall_Customers.csv"):
    try:
        df = pd.read_csv(file_path)
        X = df.iloc[:, [3, 4]].values # Annual Income, Spending Score
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler, "Mall Customers"
    except FileNotFoundError:
        return None, None, "Mall Customers (File Not Found)"

import random

class ACO:
    def __init__(self, graph, seed=None):
        self.graph = graph
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def solve(self, alpha=1.0, beta=2.0, rho=0.5, n_ants=10, n_iterations=100, plotter=None):
        nodes = list(self.graph.nodes.values())
        n_nodes = len(nodes)

        # Initialisation phéromones
        pheromone = {(i, j): 1.0 for i in range(n_nodes) for j in range(n_nodes) if i != j}

        best_path = None
        best_distance = float("inf")

        for iteration in range(n_iterations):
            all_paths = []
            all_distances = []

            for ant in range(n_ants):
                unvisited = set(range(n_nodes))
                start = random.choice(list(unvisited))
                current = start
                path = [current]
                unvisited.remove(current)

                while unvisited:
                    probabilities = []
                    next_nodes = list(unvisited)
                    for next_node in next_nodes:
                        tau = pheromone[(current, next_node)] ** alpha
                        eta = (1.0 / (self.graph.distance_function(nodes[current], nodes[next_node]) + 1e-6)) ** beta
                        probabilities.append(tau * eta)

                    # Normalisation
                    total = sum(probabilities)
                    probabilities = [p / total for p in probabilities]

                    # Tirage selon probabilité
                    r = random.random()
                    cumulative = 0
                    for idx, next_node in enumerate(next_nodes):
                        cumulative += probabilities[idx]
                        if r <= cumulative:
                            break

                    current = next_node
                    path.append(current)
                    unvisited.remove(current)

                # Retour au départ
                path.append(path[0])

                # Calcul distance
                distance = 0.0
                for i in range(len(path)-1):
                    distance += self.graph.distance_function(nodes[path[i]], nodes[path[i+1]])

                all_paths.append(path)
                all_distances.append(distance)

                # Mise à jour meilleur chemin
                if distance < best_distance:
                    best_distance = distance
                    best_path = [nodes[i] for i in path]

            # Évaporation
            for key in pheromone:
                pheromone[key] *= (1 - rho)

            # Dépôt phéromones
            for path, dist in zip(all_paths, all_distances):
                for i in range(len(path)-1):
                    pheromone[(path[i], path[i+1])] += 1.0 / (dist + 1e-6)

            # Optionnel : mise à jour plotter en temps réel
            if plotter:
                plotter.update(best_path, best_distance)

        return best_path, best_distance





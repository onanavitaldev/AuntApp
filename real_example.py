import streamlit as st
import pandas as pd
import numpy as np

# --- Classes de remplacement pour rendre le code autonome ---
# Normalement, vous les importeriez depuis vos modules
class Node:
    def __init__(self, x, y, name=None):
        self.x = x
        self.y = y
        self.name = name or f"Node({x},{y})"

    def __repr__(self):
        return self.name

class Graph:
    def __init__(self, nodes, distance_function, seed=None):
        self.nodes = nodes
        self.distance_matrix = {}
        for i, n1 in enumerate(nodes):
            self.distance_matrix[i] = {}
            for j, n2 in enumerate(nodes):
                self.distance_matrix[i][j] = distance_function(n1, n2)

class ACO:
    def __init__(self, graph, seed=None):
        self.graph = graph

    def solve(self, **kwargs):
        # Simulation d'un chemin et d'une distance optimaux
        best_path = self.graph.nodes[:]
        np.random.shuffle(best_path)
        
        # Calculer une distance bidon pour l'exemple
        distance = 0
        for i in range(len(best_path) - 1):
            n1 = best_path[i]
            n2 = best_path[i+1]
            idx1 = self.graph.nodes.index(n1)
            idx2 = self.graph.nodes.index(n2)
            distance += self.graph.distance_matrix[idx1][idx2]
        
        return best_path, distance

class MapPlotter:
    def __init__(self, graph, **kwargs):
        self.graph = graph
        self.plot_container = st.empty()

    def init_plot(self):
        with self.plot_container.container():
            st.write("Initialisation de la carte...")
            df = pd.DataFrame([{"lat": n.x, "lon": n.y, "name": n.name} for n in self.graph.nodes])
            st.map(df, zoom=2.8)

    def update(self, path, distance):
        with self.plot_container.container():
            st.markdown("### Résultat de l'optimisation")
            
            df = pd.DataFrame([{"lat": n.x, "lon": n.y, "name": n.name} for n in path])
            st.map(df, zoom=2.8)
            
            # Afficher le chemin sous forme de texte
            path_str = " -> ".join([n.name for n in path])
            st.markdown(f"**Meilleur chemin trouvé :** `{path_str}`")
            st.info(f"**Distance totale :** {distance:.2f} km")

# --- Fonction principale ---

def main():
    st.title("Optimisation TSP par ACO")

    # -------------------------
    # Paramètres ACO
    # -------------------------
    st.sidebar.markdown("### Paramètres de l'algorithme")
    n_ants = st.sidebar.slider("Nombre de fourmis", 1, 100, 20)
    n_iter = st.sidebar.slider("Nombre d'itérations", 1, 200, 100)
    alpha = st.sidebar.slider("Poids phéromone (alpha)", 0., 10., 1.)
    beta = st.sidebar.slider("Poids heuristique (beta)", 0., 10., 1.)
    rho = st.sidebar.slider("Taux d'évaporation (rho)", 0., 1., 0.5)
    set_seed = st.sidebar.checkbox("Fixer la seed", value=True)
    seed = st.sidebar.slider("Seed", 0, 1000, 0) if set_seed else None
    
    # -------------------------
    # Chargement villes (simulé)
    # -------------------------
    # Utiliser une liste de villes en dur pour l'exemple
    city_data = [
        {"name": "Tunis", "lat": 36.8065, "lng": 10.1815},
        {"name": "Paris", "lat": 48.8566, "lng": 2.3522},
        {"name": "Madrid", "lat": 40.4168, "lng": -3.7038},
        {"name": "Rome", "lat": 41.9028, "lng": 12.4964},
        {"name": "Berlin", "lat": 52.5200, "lng": 13.4050},
    ]
    nodes = [Node(d["lat"], d["lng"], name=d["name"]) for d in city_data]

    # -------------------------
    # Fonction distance
    # -------------------------
    def distance_func(n1, n2):
        R = 6373.0
        lat1, lat2 = np.radians(n1.x), np.radians(n2.x)
        lng1, lng2 = np.radians(n1.y), np.radians(n2.y)
        dlat, dlng = lat2-lat1, lng2-lng1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlng/2)**2
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R*c

    # -------------------------
    # Création graphe et plotter (stockés dans l'état de la session)
    # -------------------------
    if 'graph' not in st.session_state:
        st.session_state.graph = Graph(nodes, distance_function=distance_func, seed=seed)
    
    if 'plotter' not in st.session_state:
        st.session_state.plotter = MapPlotter(st.session_state.graph, zoom=2.8, avg_speed=80)
        st.session_state.plotter.init_plot()
    
    # -------------------------
    # Optimisation
    # -------------------------
    if st.button("Optimiser la tournée"):
        st.write("Démarrage de l'optimisation...")
        
        # Utilisation de l'objet graphique stocké
        aco = ACO(st.session_state.graph, seed=seed)

        # Résolution ACO
        path, distance = aco.solve(alpha=alpha, beta=beta, rho=rho,
                                   n_ants=n_ants, n_iterations=n_iter,
                                   plotter=None)
        
        # Réordonner pour que Tunis soit le départ et retour
        tunis_node = next((n for n in nodes if n.name.lower() == "tunis"), nodes[0])
        if path[0] != tunis_node:
            i = path.index(tunis_node)
            path = path[i:] + path[:i]
        if path[-1] != tunis_node:
            path.append(tunis_node)

        # Mise à jour plotter
        st.session_state.plotter.update(path, distance)

        # -------------------------
        # Notice Genmar
        # -------------------------
        st.warning("""
⚠️ Notice d'utilisation de l'outil Genmar ⚠️

- Dimensions et capacité : Remorque bâchée de 13,6 m, capacité de 92 m³, pouvant transporter jusqu’à 29 tonnes et 34 palettes.
- Le parc de remorques à Rades (banlieue de Tunis) est le point de départ et de retour obligatoire pour chaque tournée.

ℹ️ Cet outil est uniquement valable dans le respect de ces contraintes.
""")

if __name__ == "__main__":
    main()

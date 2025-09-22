import streamlit as st
import pandas as pd
import numpy as np
from solver.graph import Node, Graph
from solver.aco import ACO
from solver.plotter import MapPlotter

def main():
    st.title("Optimisation TSP par ACO")

    # -------------------------
    # Paramètres ACO
    # -------------------------
    n_ants = st.sidebar.slider("Nombre de fourmis", 1, 100, 20)
    n_iter = st.sidebar.slider("Nombre d'itérations", 1, 200, 100)
    alpha = st.sidebar.slider("Poids phéromone (alpha)", 0., 10., 1.)
    beta = st.sidebar.slider("Poids heuristique (beta)", 0., 10., 1.)
    rho = st.sidebar.slider("Taux d'évaporation (rho)", 0., 1., 0.5)
    set_seed = st.sidebar.checkbox("Fixer la seed", value=True)
    seed = st.sidebar.slider("Seed", 0, 1000, 0) if set_seed else None

    # -------------------------
    # Chargement villes
    # -------------------------
    df = pd.read_csv("data/european_cities.csv")
    nodes = []
    for _, row in df.iterrows():
        try:
            nodes.append(Node(float(row["lat"]), float(row["lng"]), name=row["name"]))
        except:
            continue

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
    # Création graphe
    # -------------------------
    graph = Graph(nodes, distance_function=distance_func, seed=seed)

    # -------------------------
    # Plotter
    # -------------------------
    plotter = MapPlotter(graph, zoom=2.8, avg_speed=80)
    plotter.init_plot()  # crée la carte

    # -------------------------
    # Optimisation dynamique
    # -------------------------
    if st.button("Optimiser la tournée"):
        aco = ACO(graph, seed=seed)

        # Résolution ACO
        path, distance = aco.solve(alpha=alpha, beta=beta, rho=rho,
                                   n_ants=n_ants, n_iterations=n_iter,
                                   plotter=None)  # plotter = None pour mise à jour finale

        # Réordonner pour que Tunis soit le départ et retour
        tunis_node = next((n for n in nodes if n.name.lower() == "tunis"), nodes[0])
        if path[0] != tunis_node:
            i = path.index(tunis_node)
            path = path[i:] + path[:i]
        if path[-1] != tunis_node:
            path.append(tunis_node)  # fermer la boucle

        # Mise à jour plotter
        plotter.update(path, distance)

        # -------------------------
        # Notice Genmar en bas
        # -------------------------
        st.warning("""
⚠️ Notice d'utilisation de l'outil Genmar ⚠️

- Dimensions et capacité : Remorque bâchée de 13,6 m, capacité de 92 m³, pouvant transporter jusqu’à 29 tonnes et 34 palettes.
- Le parc de remorques à Rades (banlieue de Tunis) est le point de départ et de retour obligatoire pour chaque tournée.

ℹ️ Cet outil est uniquement valable dans le respect de ces contraintes.
""")

if __name__ == "__main__":
    main()

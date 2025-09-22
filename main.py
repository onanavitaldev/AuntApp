import streamlit as st
import pandas as pd
import numpy as np
from solver.graph import Node, Graph
from solver.aco import ACO
from solver.plotter import MapPlotter

# ---- Personnalisation CSS ----
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #2C3E50;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 15px;
        border-radius: 12px;
        background: linear-gradient(135deg, #D6EAF8, #85C1E9);
        border: 1px solid #2980B9;
        margin-top: 15px;
        color: #1B2631;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #1B4F72;
        color: #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    # ---- Titre principal ----
    st.markdown("<div class='main-title'>📍 Cartographie de la tournée</div>", unsafe_allow_html=True)

    # ---- Paramètres ACO ----
    st.sidebar.title("⚙️ Paramètres de l’optimisation")
    n_ants = st.sidebar.slider("Nombre de fourmis", 1, 100, 20)
    n_iter = st.sidebar.slider("Nombre d'itérations", 1, 200, 100)
    alpha = st.sidebar.slider("Poids phéromone (alpha)", 0., 10., 1.)
    beta = st.sidebar.slider("Poids heuristique (beta)", 0., 10., 1.)
    rho = st.sidebar.slider("Taux d'évaporation (rho)", 0., 1., 0.5)
    set_seed = st.sidebar.checkbox("Fixer la seed", value=True)
    seed = st.sidebar.slider("Seed", 0, 1000, 0) if set_seed else None

    # ---- Chargement des villes ----
    df = pd.read_csv("data/european_cities.csv")
    nodes = []
    for _, row in df.iterrows():
        try:
            nodes.append(Node(float(row["lat"]), float(row["lng"]), name=row["name"]))
        except:
            continue

    # ---- Fonction distance (Haversine) ----
    def distance_func(n1, n2):
        R = 6373.0
        lat1, lat2 = np.radians(n1.x), np.radians(n2.x)
        lng1, lng2 = np.radians(n1.y), np.radians(n2.y)
        dlat, dlng = lat2-lat1, lng2-lng1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlng/2)**2
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R*c

    # ---- Création du graphe ----
    graph = Graph(nodes, distance_function=distance_func, seed=seed)

    # ---- Plotter ----
    plotter = MapPlotter(graph, zoom=5.0)
    plotter.init_plot()

    # ---- Interface principale ----
    if st.button("🚀 Optimiser la tournée"):
        aco = ACO(graph, seed=seed)
        path, _ = aco.solve(alpha=alpha, beta=beta, rho=rho,
                            n_ants=n_ants, n_iterations=n_iter, plotter=None)

        # ---- Nettoyage du chemin pour éviter les répétitions ----
        tunis_node = next((n for n in nodes if n.name.lower() == "tunis"), nodes[0])
        seen = set()
        final_path = [tunis_node]
        for n in path:
            if n.name not in seen and n != tunis_node:
                final_path.append(n)
                seen.add(n.name)
        final_path.append(tunis_node)  # retour à Tunis

        # ---- Calcul distance totale ----
        total_distance = 0
        for i in range(len(final_path)-1):
            total_distance += distance_func(final_path[i], final_path[i+1])

        # ---- Mise à jour du plotter ----
        plotter.update(final_path, total_distance)

        # ---- Affichage des résultats ----
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.subheader("📊 Résultats de l’optimisation")
        st.write("**Itinéraire optimal :**")
        st.write(" → ".join([n.name for n in final_path]))
        st.write(f"**Distance totale :** {total_distance:.2f} km")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Notice en bas ----
    st.info("""
⚠️ Cet outil est valable uniquement pour les contraintes de l’entreprise Genmar :
- Dimensions et capacité : Remorque bâchée de 13,6 m, capacité de 92 m³, jusqu’à 29 tonnes et 34 palettes  
- Le parc de remorques à Radès est le point de départ et de retour obligatoire pour chaque tournée
    """)


if __name__ == "__main__":
    main()




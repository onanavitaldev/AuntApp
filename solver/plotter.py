import streamlit as st
import pandas as pd
import pydeck

class MapPlotter:
    def __init__(self, graph, zoom=3.0):
        self.graph = graph
        self.zoom = zoom
        self.r = None
        self.layer_pheromones = None
        self.layer_best_path = None
        self.chart = None
        self.df_distance = None
        self.distance_chart = None
        self.best_distance = float("inf")
        self.worst_distance = 0.0

    def init_plot(self):
        # Nodes
        nodes_name = [{"coordinates": [n.y, n.x], "name": n.name} for n in self.graph.nodes.values()]
        layer_nodes = pydeck.Layer(
            "ScatterplotLayer",
            nodes_name,
            pickable=True,
            get_position="coordinates",
            radius_min_pixels=5,
            get_color=[0, 166, 251],
        )

        # Pheromones en noir
        self.layer_pheromones = pydeck.Layer(
            "LineLayer",
            self._get_lines_pheromones(),
            get_source_position="start",
            get_target_position="end",
            get_width="value",
            width_scale=4,
            pickable=True,
            get_color=[50, 50, 50],
        )

        # Trajet optimal (rouge)
        self.layer_best_path = pydeck.Layer(
            "LineLayer",
            [],
            get_source_position="start",
            get_target_position="end",
            get_color="color",
            width_scale=3,
            pickable=True,
        )

        # Deck
        init_view = self._get_init_view(nodes_name)
        self.r = pydeck.Deck(
            layers=[layer_nodes, self.layer_pheromones, self.layer_best_path],
            initial_view_state=init_view,
            map_style="",
            tooltip={"text": "name"}
        )
        self.chart = st.pydeck_chart(self.r)

        # Convergence chart
        self.df_distance = pd.DataFrame({"Best distance": []})
        self.distance_chart = st.line_chart(self.df_distance)

    def update(self, best_path, distance):
        if not best_path or len(best_path) < 2:
            return

        # Supprimer doublons tout en gardant ordre
        seen = set()
        unique_path = []
        for n in best_path:
            if n.name not in seen:
                unique_path.append(n)
                seen.add(n.name)
        best_path = unique_path
        # Fermer la boucle
        if best_path[0] != best_path[-1]:
            best_path.append(best_path[0])

        self.best_distance = min(self.best_distance, distance)
        self.worst_distance = max(self.worst_distance, distance)

        # Trajet optimal en rouge
        lines_best_path = []
        for i in range(len(best_path)-1):
            lines_best_path.append({
                "start": [best_path[i].y, best_path[i].x],
                "end": [best_path[i+1].y, best_path[i+1].x],
                "color": [255, 0, 0]
            })

        self.layer_best_path.data = lines_best_path
        self.layer_pheromones.data = self._get_lines_pheromones()
        self.chart.pydeck_chart = self.r

        # Convergence chart
        self.df_distance = pd.concat([self.df_distance, pd.DataFrame({"Best distance": [distance]})], ignore_index=True)
        self.distance_chart.line_chart(self.df_distance)

    def _get_init_view(self, nodes):
        if not nodes:
            return pydeck.ViewState(latitude=0, longitude=0, zoom=self.zoom)
        latitudes = [n["coordinates"][1] for n in nodes]
        longitudes = [n["coordinates"][0] for n in nodes]
        return pydeck.ViewState(
            latitude=(max(latitudes) + min(latitudes)) / 2,
            longitude=(max(longitudes) + min(longitudes)) / 2,
            zoom=self.zoom
        )

    def _get_lines_pheromones(self):
        lines = []
        ph = self.graph.retrieve_pheromone()
        for (i, j), val in ph.items():
            n1, n2 = self.graph.nodes[i], self.graph.nodes[j]
            lines.append({
                "start": [n1.y, n1.x],
                "end": [n2.y, n2.x],
                "value": float(val)
            })
        return lines




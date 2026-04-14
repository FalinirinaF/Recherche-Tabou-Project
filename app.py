import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import random
from copy import deepcopy

# ===============================
# CONFIGURATION DE LA PAGE
# ===============================
st.set_page_config(page_title="VRP Tabou Pro - Simulation", layout="wide")

# ===============================
# FONCTIONS TECHNIQUES (BACKEND)
# ===============================

def compute_distance_matrix(df):
    coords = df[['x', 'y']].values
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(coords[i] - coords[j])
    return dist

def route_distance(route, dist_matrix):
    return sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route) - 1))

def total_distance(solution, dist_matrix):
    return sum(route_distance(route, dist_matrix) for route in solution)

# --- Initialisation Gloutonne optimisée pour répartir sur les camions dispos ---
def greedy_initial_solution(dist_matrix, demands, Q, max_trucks):
    n = len(demands)
    customers = list(range(1, n))
    # On initialise le nombre de routes demandé
    solution = [[0, 0] for _ in range(max_trucks)]
    
    curr_truck = 0
    while customers:
        # On essaie d'ajouter le client le plus proche au camion actuel
        current_pos = solution[curr_truck][-2]
        
        # Vérification de la capacité pour le camion actuel
        current_load = sum(demands[c] for c in solution[curr_truck])
        feasible = [c for c in customers if current_load + demands[c] <= Q]
        
        if feasible:
            # On prend le plus proche pour garder une logique gloutonne
            next_c = min(feasible, key=lambda c: dist_matrix[current_pos][c])
            solution[curr_truck].insert(-1, next_c)
            customers.remove(next_c)
        
        # On passe au camion suivant pour forcer la répartition sur toute la flotte
        curr_truck = (curr_truck + 1) % max_trucks
        
        # Sécurité : si un client ne rentre nulle part, on le force dans le moins chargé
        if not feasible and customers and all(sum(demands[c] for c in r) + demands[customers[0]] > Q for r in solution):
            loads = [sum(demands[c] for c in r) for r in solution]
            min_truck = loads.index(min(loads))
            solution[min_truck].insert(-1, customers.pop(0))

    return [r for r in solution if len(r) > 2]

# --- Moteur Recherche Tabou ---
def tabu_search(initial_sol, dist_matrix, demands, Q, tabu_size, max_iter):
    best_sol = deepcopy(initial_sol)
    best_cost = total_distance(best_sol, dist_matrix)
    curr_sol = deepcopy(initial_sol)
    tabu_list, history = [], [best_cost]

    for _ in range(max_iter):
        neighbors = []
        for i in range(len(curr_sol)):
            if len(curr_sol[i]) <= 2: continue
            for j in range(1, len(curr_sol[i])-1):
                new_sol = deepcopy(curr_sol)
                customer = new_sol[i].pop(j)
                dest_route = random.randint(0, len(new_sol)-1)
                
                if sum(demands[c] for c in new_sol[dest_route]) + demands[customer] <= Q:
                    new_sol[dest_route].insert(1, customer)
                    neighbors.append((new_sol, ("relocate", customer, dest_route)))

        if not neighbors: break
        neighbors.sort(key=lambda x: total_distance(x[0], dist_matrix))
        
        for sol, move in neighbors:
            cost = total_distance(sol, dist_matrix)
            if move not in tabu_list or cost < best_cost:
                curr_sol = sol
                if cost < best_cost:
                    best_sol, best_cost = deepcopy(sol), cost
                tabu_list.append(move)
                if len(tabu_list) > tabu_size: tabu_list.pop(0)
                break
        history.append(best_cost)
    return best_sol, history

# ===============================
# VISUALISATION (FRONTEND)
# ===============================

def create_plotly_fig(df, solution, step, title):
    fig = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
    
    for idx, route in enumerate(solution):
        visible_route = route[:step+1]
        route_df = df.iloc[visible_route]
        
        fig.add_trace(go.Scatter(
            x=route_df['x'], y=route_df['y'],
            mode='markers+lines',
            name=f'Camion {idx+1}',
            line=dict(color=colors[idx % len(colors)], width=3),
            marker=dict(size=10),
            text=[f"Client {r}<br>Demande: {df.iloc[r]['demand']}" for r in visible_route],
            hoverinfo='text'
        ))
    
    fig.add_trace(go.Scatter(
        x=[df.iloc[0]['x']], y=[df.iloc[0]['y']],
        mode='markers',
        marker=dict(symbol='square', size=18, color='red'),
        name='Dépôt'
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(range=[-5, 105], title="Coord X"),
        yaxis=dict(range=[-5, 105], title="Coord Y"),
        height=550,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ===============================
# APPLICATION PRINCIPALE
# ===============================

st.title("🚀 Optimisation de Tournées (VRP) avec Recherche Tabou")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # MODIFICATION ICI : Utilisation de number_input pour les boutons +/-
    n_trucks = st.number_input(
        "Nombre de camions disponibles", 
        min_value=1, 
        max_value=10, 
        value=3, 
        step=1
    )
    
    n_cust = st.slider("Nombre de clients", 5, 40, 15)
    cap = st.slider("Capacité des camions", 10, 80, 35)
    
    st.header("🧠 Paramètres Algorithme")
    t_size = st.slider("Taille liste Tabou", 5, 50, 15)
    iters = st.slider("Itérations Max", 50, 500, 150)
    
    st.header("🎞️ Animation")
    speed = st.select_slider("Vitesse du trajet", options=["Très Lent", "Lent", "Normal", "Rapide"], value="Normal")
    sleep_time = {"Très Lent": 1.5, "Lent": 1.0, "Normal": 0.4, "Rapide": 0.05}[speed]
    
    run = st.button("🚀 Lancer l'Optimisation")
    st.button("🔄 Nouveaux clients", on_click=lambda: st.session_state.pop('df', None))

if 'df' not in st.session_state:
    data = {'x': np.random.rand(n_cust+1)*100, 'y': np.random.rand(n_cust+1)*100, 
            'demand': np.random.randint(2, 10, size=n_cust+1)}
    st.session_state.df = pd.DataFrame(data)
    st.session_state.df.iloc[0, 2] = 0 

if run:
    df = st.session_state.df
    dist_m = compute_distance_matrix(df)
    
    init_sol = greedy_initial_solution(dist_m, df['demand'].values, cap, n_trucks)
    best_sol, history = tabu_search(init_sol, dist_m, df['demand'].values, cap, t_size, iters)
    
    d_init = total_distance(init_sol, dist_m)
    d_best = total_distance(best_sol, dist_m)
    gain = ((d_init - d_best) / d_init) * 100

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Distance Initiale", f"{d_init:.2f} km")
    col_m2.metric("Distance Optimisée", f"{d_best:.2f} km")
    col_m3.metric("Amélioration", f"{gain:.1f}%", delta=f"{gain:.1f}%")

    st.subheader("🚛 Simulation du mouvement des camions")
    graph_col1, graph_col2 = st.columns(2)
    p1, p2 = graph_col1.empty(), graph_col2.empty()

    max_steps = max(len(r) for r in init_sol + best_sol)
    
    for s in range(1, max_steps + 1):
        p1.plotly_chart(create_plotly_fig(df, init_sol, s, "Solution Gloutonne"), use_container_width=True, key=f"init_{s}")
        p2.plotly_chart(create_plotly_fig(df, best_sol, s, f"Solution Tabou ({len(best_sol)} camions)"), use_container_width=True, key=f"best_{s}")
        time.sleep(sleep_time)

    st.success("✅ Livraison terminée !")
    
    st.divider()
    st.subheader("📊 Rapport d'activité par Camion")
    report = []
    for i, r in enumerate(best_sol):
        report.append({
            "Camion": i + 1,
            "Clients": len(r) - 2,
            "Charge (kg)": sum(df.iloc[r]['demand']),
            "Distance (km)": round(route_distance(r, dist_m), 2)
        })
    st.table(pd.DataFrame(report))

    st.subheader("📉 Évolution de la performance")
    st.line_chart(history)

    total_reduction = d_init - d_best
    last_imp = 0
    for i in range(1, len(history)):
        if history[i] < history[i-1]: last_imp = i

    st.info("💡 **Interprétation des résultats :**")
    c_a, c_b = st.columns(2)
    with c_a:
        st.write(f"**Analyse de distance :**")
        st.write(f"- Réduction totale : **{total_reduction:.2f} km**")
        st.write(f"- L'algorithme a réduit le trajet de **{gain:.1f}%**.")
    with c_b:
        st.write(f"**Analyse de convergence :**")
        st.write(f"- Stabilisation atteinte à l'itération : **{last_imp}**")
        status = "rapide" if last_imp < iters/2 else "progressive"
        st.write(f"- La convergence a été **{status}** sur ce scénario.")

    st.write(f"**Conclusion :** La Recherche Tabou a optimisé les tournées en respectant la limite de **{n_trucks} camions**. "
             "Si vous choisissez un seul camion, le problème devient un TSP. Avec plusieurs camions, l'algorithme gère la répartition complexe du VRP.")

else:
    st.info("Ajustez les paramètres et cliquez sur 'Lancer l'Optimisation'.")

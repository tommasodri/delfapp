import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns

# Dati delle posizioni
posizioni = {
    "Alice C":   [12, 13,  9,  4,  2,  6,  4,  3,  6,  4,  7,  5, 12],
    "Alice P":   [ 3,  8, 13, None,  7, None,  5,  9, None, 13,  8, None, None],
    "Assia":     [ 9,  5,  4,  3,  5,  5,  7, 12, 11, 11,  5,  7,  5],
    "Aurora":    [14,  3, 11,  6,  8,  4, 12, None, 10,  6, None,  6,  9],
    "Chiara":    [11, 10, 12, 11, 15, None,  2,  2, None, None, None, 11, None],
    "Delfina":   [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    "Diletta":   [ 4,  6,  2,  2, 12,  2, None,  5,  2,  3, 10,  8,  3],
    "Elena":     [13, 12,  7, None,  9, None,  9, 14,  3, 10, None,  9,  7],
    "Eleonora":  [ 8,  2,  3, 13,  6,  3, 11,  4, 12,  2,  9, 10, 11],
    "Elisa C":   [ 5,  8,  5,  8, 10,  7, 10, 11,  4,  7,  3,  3,  4],
    "Elisa M":   [ 2,  4,  6, 10, 14, 10, None, 13, None, 12, None, None,  6],
    "Marta":     [15,  9, None,  5,  4,  8,  3, 10,  9, 14,  2, None,  8],
    "Silvia":    [10,  7,  8, 12, 11, None,  8,  8,  8,  9,  4, None, None],
    "Valentina": [ 6, 14, 10,  9, 13,  9,  6,  7,  7,  8,  6,  2, 10],
    "Zaira":     [ 7, 11, None,  7,  3, 11, 13,  6,  5,  5, None,  4,  2],
}

colori_nodi = {
    "Alice C": "#800000",        # bordeaux
    "Alice P": "#228B22",        # verde bosco
    "Assia": "#FFD700",          # oro
    "Aurora": "#FFB6C1",         # rosa chiaro
    "Chiara": "#C0C0C0",         # argento
    "Delfina": "#FFFFFF",        # bianco
    "Diletta": "#D40078",        # rosa ciclamino
    "Elena": "#0000FF",          # blu
    "Eleonora": "#191970",       # blu notte
    "Elisa C": "#8A2BE2",        # viola
    "Elisa M": "#FFA500",        # arancione
    "Marta": "#87CEEB",          # celeste
    "Silvia": "#FF0000",         # rosso
    "Valentina": "#008000",      # verde
    "Zaira": "#FF69B4",          # rosa
}

partecipanti = list(posizioni.keys())
n_incontri = 13
incontri = range(n_incontri)

# Funzione per calcolare la distanza in un cerchio
def distanza_cerchio(pos1, pos2, N):
    return min(abs(pos1 - pos2), N - abs(pos1 - pos2))

def calcola_matrice_vicinanza(incontro):
    posizioni_presenti = [posizioni[p][incontro] for p in partecipanti if posizioni[p][incontro] is not None]
    N = max(posizioni_presenti)
    matrice_vicinanza = pd.DataFrame(np.zeros((len(partecipanti), len(partecipanti))), index=partecipanti, columns=partecipanti)
    for i, p1 in enumerate(partecipanti):
        for j, p2 in enumerate(partecipanti):
            if posizioni[p1][incontro] is not None and posizioni[p2][incontro] is not None:
                matrice_vicinanza.loc[p1, p2] = distanza_cerchio(posizioni[p1][incontro], posizioni[p2][incontro], N)
    return matrice_vicinanza

# Funzione per calcolare la matrice di vicinanza media
def calcola_matrice_vicinanza_media():
    matrici = []
    for incontro in incontri:
        matrici.append(calcola_matrice_vicinanza(incontro))
    matrice_vicinanza_media = sum(matrici) / len(matrici)
    return matrice_vicinanza_media

# Funzione per determinare se un colore è chiaro o scuro
def is_color_light(color_hex):
    color = color_hex.lstrip('#')
    r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
    # Utilizzare la formula della luminanza percepita
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return luminance > 0.5

def print_heatmap(mask, fig, ax, matrice_vicinanza):
    # Visualizza la matrice di vicinanza con una mappa di calore
    ax = sns.heatmap(matrice_vicinanza, annot=True, fmt=".0f", cmap="RdYlGn_r", ax=ax, mask=mask, cbar_kws={"label": "Distanza"})

    # Sposta le etichette dell'asse X in alto
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=90)

    st.pyplot(fig)

def print_graph(G, fig, ax, pos, partecipanti_presenti):
    # Aggiunta dei nodi
    for partecipante in partecipanti_presenti:
        G.add_node(partecipante, color=colori_nodi[partecipante])

    # Disegno dei nodi con i colori specificati
    node_colors = [nx.get_node_attributes(G, 'color')[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, ax=ax)

    # Draw edges with colors based on weights
    edges = G.edges(data=True)
    weights = np.array([d['weight'] for u, v, d in edges])
    norm_weights = (weights - weights.min()) / (weights.max() - weights.min())
    edge_colors = [plt.cm.RdYlGn_r(1 - norm_weight) for norm_weight in norm_weights]

    for (u, v, d), color in zip(edges, edge_colors):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=d['weight'], edge_color=[color], ax=ax)

    # Draw labels with contrasting colors
    for node, (x, y) in pos.items():
        node_color = colori_nodi[node]
        color = 'black' if is_color_light(node_color) else 'white'
        ax.text(x, y, node, fontsize=10, ha='center', va='center', color=color, fontweight='bold')

    st.pyplot(fig)

# Configurazione dell'app Streamlit
st.title("Analisi delle vicinanze nei vari incontri")
incontro_selezionato = st.slider("Seleziona l'incontro", 1, n_incontri, 1) - 1
tipo_grafo = st.selectbox("Seleziona il tipo di grafo", ["Heatmap", "Network Graph"])

matrice_vicinanza = calcola_matrice_vicinanza(incontro_selezionato)

if tipo_grafo == "Heatmap":
    mask = matrice_vicinanza == 0
    fig, ax = plt.subplots()
    # plt.title(f'Matrice di vicinanza per l\'incontro {incontro_selezionato + 1}')
    print_heatmap(mask, fig, ax, matrice_vicinanza)
else:
    G = nx.Graph()

    partecipanti_presenti = [p for p in partecipanti if posizioni[p][incontro_selezionato] is not None]

    for i in range(len(partecipanti_presenti)):
        for j in range(i + 1, len(partecipanti_presenti)):
            p1, p2 = partecipanti_presenti[i], partecipanti_presenti[j]
            if matrice_vicinanza.loc[p1, p2] > 0:
                G.add_edge(p1, p2, weight=matrice_vicinanza.loc[p1, p2])

    pos = nx.spring_layout(G)
    edge_weights = nx.get_edge_attributes(G, 'weight')

    fig, ax = plt.subplots(figsize=(14, 10))

    print_graph(G, fig, ax, pos, partecipanti_presenti)

st.divider()

st.title("Analisi delle vicinanze medie su tutti gli incontri")

matrice_vicinanza_media = calcola_matrice_vicinanza_media()

st.subheader("Heatmap delle distanze medie")
mask = matrice_vicinanza_media == 0
fig, ax = plt.subplots()
plt.title('Matrice di vicinanza media')
print_heatmap(mask, fig, ax, matrice_vicinanza_media)

st.subheader("Grafo delle distanze medie")
G = nx.Graph()

partecipanti_presenti = [p for p in partecipanti if any(posizioni[p])]

for i in range(len(partecipanti_presenti)):
    for j in range(i + 1, len(partecipanti_presenti)):
        p1, p2 = partecipanti_presenti[i], partecipanti_presenti[j]
        if matrice_vicinanza_media.loc[p1, p2] > 0:
            G.add_edge(p1, p2, weight=matrice_vicinanza_media.loc[p1, p2])

pos = nx.spring_layout(G)
edge_weights = nx.get_edge_attributes(G, 'weight')

fig, ax = plt.subplots(figsize=(14, 10))

print_graph(G, fig, ax, pos, partecipanti_presenti)
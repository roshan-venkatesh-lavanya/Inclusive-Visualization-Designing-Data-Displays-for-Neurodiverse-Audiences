import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go


# ---------------------------
# 1. Load MovieLens + NRC-VAD
# ---------------------------

MOVIES_PATH = "movies.csv"
RATINGS_PATH = "ratings.csv"
TAGS_PATH = "tags.csv"

# IMPORTANT: update this to match your NRC-VAD file name & column names.
# Expected columns after renaming: word, valence, arousal, dominance
VAD_PATH = "NRC-VAD-Lexicon-v2.1.txt"

movies = pd.read_csv(MOVIES_PATH)
ratings = pd.read_csv(RATINGS_PATH)
tags = pd.read_csv(TAGS_PATH)

# Load NRC-VAD lexicon
# Try multiple loading strategies to handle different file formats
try:
    # Strategy 1: Tab-separated with actual header
    vad_raw = pd.read_csv(VAD_PATH, sep='\t', header=0)
    
    print(f"Original columns: {vad_raw.columns.tolist()}")
    
    # Standardize column names (handle variations in the file)
    vad_raw.columns = vad_raw.columns.str.lower().str.strip()
    
    print(f"Lowercase columns: {vad_raw.columns.tolist()}")
    
    # Map to expected column names
    column_mapping = {}
    for col in vad_raw.columns:
        if 'word' in col or 'term' in col: 
            column_mapping[col] = 'word'
        elif 'valence' in col:
            column_mapping[col] = 'valence'
        elif 'arousal' in col:
            column_mapping[col] = 'arousal'
        elif 'dominance' in col or 'dominan' in col:
            column_mapping[col] = 'dominance'
    
    print(f"Column mapping: {column_mapping}")
    
    if column_mapping:
        vad_raw = vad_raw.rename(columns=column_mapping)
    else:
        # No mapping found, assume first 4 columns are word, valence, arousal, dominance
        vad_raw.columns = ['word', 'valence', 'arousal', 'dominance'] + list(vad_raw.columns[4:])
    
    print(f"Final columns: {vad_raw.columns.tolist()}")
    
    # Keep only the columns we need
    vad_raw = vad_raw[['word', 'valence', 'arousal', 'dominance']]
    
except Exception as e:
    print(f"Strategy 1 failed: {e}")
    try:
        # Strategy 2: Read as whitespace-separated, skip problematic rows
        vad_raw = pd.read_csv(
            VAD_PATH,
            sep=r'\s+',
            header=0,
            engine='python',
            on_bad_lines='skip'  # Skip malformed lines
        )
        print(f"Strategy 2 columns: {vad_raw.columns.tolist()}")
        vad_raw.columns = ['word', 'valence', 'arousal', 'dominance'] + list(vad_raw.columns[4:])
        vad_raw = vad_raw[['word', 'valence', 'arousal', 'dominance']]
    except Exception as e2:
        print(f"Strategy 2 failed: {e2}")
        # Strategy 3: Manual parsing
        with open(VAD_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data = []
        for i, line in enumerate(lines[1:], 1):  # Skip header
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                try:
                    data.append({
                        'word': parts[0],
                        'valence': float(parts[1]),
                        'arousal': float(parts[2]),
                        'dominance': float(parts[3])
                    })
                except (ValueError, IndexError):
                    continue
        
        vad_raw = pd.DataFrame(data)
        print(f"Strategy 3 loaded {len(vad_raw)} rows")

# Clean the word column
vad_raw["word"] = vad_raw["word"].str.lower().str.strip()

# Convert numeric columns
vad_raw["valence"] = pd.to_numeric(vad_raw["valence"], errors="coerce")
vad_raw["arousal"] = pd.to_numeric(vad_raw["arousal"], errors="coerce")
vad_raw["dominance"] = pd.to_numeric(vad_raw["dominance"], errors="coerce")

vad_raw = vad_raw.dropna(subset=["valence", "arousal", "dominance"])

print(f"Loaded {len(vad_raw)} words from NRC-VAD lexicon")
# ---------------------------
# 2. Aggregate VAD per movie
# ---------------------------

# Normalize / clean tags
tags_clean = tags.copy()
tags_clean["tag_norm"] = tags_clean["tag"].str.lower().str.strip()

# Join tags with VAD lexicon
tags_vad = tags_clean.merge(
    vad_raw,
    left_on="tag_norm",
    right_on="word",
    how="inner"
)

# Aggregate VAD scores per movie (average of all matching tags)
movie_vad = (
    tags_vad
    .groupby("movieId")[["valence", "arousal", "dominance"]]
    .mean()
    .reset_index()
)

# Average rating per movie – proxy for audience engagement
movie_rating = (
    ratings
    .groupby("movieId")["rating"]
    .mean()
    .reset_index()
    .rename(columns={"rating": "mean_rating"})
)

# Merge everything into a single dataframe
movies_vad = (
    movies
    .merge(movie_vad, on="movieId", how="inner")
    .merge(movie_rating, on="movieId", how="left")
)

# Drop rows with missing values (if any)
movies_vad = movies_vad.dropna(subset=["valence", "arousal", "dominance"])

# Emotional intensity: distance from neutral (0.5, 0.5) in valence-arousal space
# Adjust if your VAD scale is [1,9] etc. For 0–1 scale, this is fine.
# For 1–9, you might want to divide by 9.
movies_vad["emotional_intensity"] = np.sqrt(
    (movies_vad["valence"] - 0.5) ** 2 + (movies_vad["arousal"] - 0.5) ** 2
)

# Normalize valence for color mapping
movies_vad["valence_norm"] = (
    (movies_vad["valence"] - movies_vad["valence"].min())
    / (movies_vad["valence"].max() - movies_vad["valence"].min())
)

# Normalize rating for node size
movies_vad["rating_norm"] = (
    (movies_vad["mean_rating"] - movies_vad["mean_rating"].min())
    / (movies_vad["mean_rating"].max() - movies_vad["mean_rating"].min())
)
movies_vad["node_size"] = 20 + movies_vad["rating_norm"] * 25  # 20–45

# ---------------------------
# 3. Similarity matrix (VAD-based)
# ---------------------------

# We'll use cosine similarity over [valence, arousal, dominance]
vad_features = movies_vad[["valence", "arousal", "dominance"]].values
sim_matrix = cosine_similarity(vad_features)

# Map between index in movies_vad and movieId
idx_to_movieId = movies_vad["movieId"].values
movieId_to_idx = {mid: i for i, mid in enumerate(idx_to_movieId)}


# ---------------------------
# 4. Network layout helper
# ---------------------------

def get_neighbors(center_movie_id, mode="intense", max_neighbors_intense=8, max_neighbors_simplified=4):
    """Return a list of neighbor movieIds sorted by similarity."""
    if center_movie_id not in movieId_to_idx:
        # Fallback to first movie in the VAD-filtered set
        center_movie_id = idx_to_movieId[0]

    center_idx = movieId_to_idx[center_movie_id]
    sims = sim_matrix[center_idx]

    # Exclude self, sort by similarity descending
    neighbor_indices = np.argsort(-sims)
    neighbor_indices = [i for i in neighbor_indices if i != center_idx]

    if mode == "intense":
        k = max_neighbors_intense
    else:
        k = max_neighbors_simplified

    neighbor_indices = neighbor_indices[:k]
    neighbor_movieIds = [idx_to_movieId[i] for i in neighbor_indices]
    return neighbor_movieIds


def build_radial_layout(center_movie_id, neighbor_movieIds, radius=1.0):
    """
    Return a dict: movieId -> (x, y)
    Center movie is at (0, 0), neighbors on a circle.
    """
    positions = {}
    positions[center_movie_id] = (0.0, 0.0)

    n = len(neighbor_movieIds)
    if n == 0:
        return positions

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    for movie_id, angle in zip(neighbor_movieIds, angles):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions[movie_id] = (x, y)

    return positions


def movie_info(movie_id):
    row = movies_vad.loc[movies_vad["movieId"] == movie_id].iloc[0]
    return {
        "movieId": int(row["movieId"]),
        "title": row["title"],
        "genres": row["genres"],
        "valence": float(row["valence"]),
        "arousal": float(row["arousal"]),
        "dominance": float(row["dominance"]),
        "intensity": float(row["emotional_intensity"]),
        "rating": float(row["mean_rating"]),
    }


# ---------------------------
# 5. Build Plotly network figure
# ---------------------------

CALM_COLORSCALE = [
    [0.0, "#6B9BD1"],  # low valence → cool blue
    [0.5, "#A8D5BA"],  # neutral → soft green
    [1.0, "#F7B267"],  # high valence → warm muted orange
]

def build_network_figure(center_movie_id, mode="intense"):
    if center_movie_id not in movieId_to_idx:
        center_movie_id = idx_to_movieId[0]

    neighbor_movieIds = get_neighbors(center_movie_id, mode=mode)
    positions = build_radial_layout(center_movie_id, neighbor_movieIds, radius=1.0)

    # Collect node info
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    node_customdata = []

    for movie_id, (x, y) in positions.items():
        row = movies_vad.loc[movies_vad["movieId"] == movie_id].iloc[0]
        node_x.append(x)
        node_y.append(y)
        node_text.append(row["title"])
        node_size.append(row["node_size"])
        node_color.append(row["valence_norm"])
        node_customdata.append(int(movie_id))

    # Edges: from center to each neighbor
    edge_x = []
    edge_y = []
    center_x, center_y = positions[center_movie_id]

    for movie_id in neighbor_movieIds:
        x0, y0 = center_x, center_y
        x1, y1 = positions[movie_id]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2, color="rgba(150,150,150,0.5)"),
        hoverinfo="none",
        showlegend=False,
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[t if mode == "intense" else "" for t in node_text],
        textposition="top center",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale=CALM_COLORSCALE,
            cmin=0,
            cmax=1,
            line=dict(width=3, color="white"),
        ),
        hoverinfo="text",
        hovertext=[
            f"{movies_vad.loc[movies_vad['movieId'] == mid].iloc[0]['title']}<br>"
            f"Genres: {movies_vad.loc[movies_vad['movieId'] == mid].iloc[0]['genres']}<br>"
            f"Mean rating: {movies_vad.loc[movies_vad['movieId'] == mid].iloc[0]['mean_rating']:.2f}<br>"
            f"Valence: {movies_vad.loc[movies_vad['movieId'] == mid].iloc[0]['valence']:.3f}<br>"
            f"Arousal: {movies_vad.loc[movies_vad['movieId'] == mid].iloc[0]['arousal']:.3f}<br>"
            f"Dominance: {movies_vad.loc[movies_vad['movieId'] == mid].iloc[0]['dominance']:.3f}"
            for mid in positions.keys()
        ],
        customdata=node_customdata,  # used to know which movie was clicked
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        template="simple_white",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=600,
        plot_bgcolor="rgba(250,250,252,1)",
        paper_bgcolor="rgba(250,250,252,1)",
        title="Emotional Similarity Network (NRC-VAD + MovieLens)",
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)  # keep aspect ratio circular

    return fig


# ---------------------------
# 6. Dash layout
# ---------------------------

# Default center movie: first in movies_vad
default_center_movie_id = int(movies_vad.iloc[0]["movieId"])

app = Dash(__name__)

app.layout = html.Div(
    style={
        "fontFamily": "'Segoe UI', system-ui, sans-serif",
        "background": "linear-gradient(135deg, #f3f6fb, #eef9f4)",
        "minHeight": "100vh",
        "padding": "20px",
    },
    children=[
        html.H2(
            "Inclusive Emotional Movie Explorer",
            style={"textAlign": "center", "marginBottom": "10px"},
        ),
        html.P(
            "Explore movies positioned by emotional similarity using NRC-VAD scores. "
            "Click a node to recenter, hover to see details, and adjust the view mode "
            "for different cognitive loads.",
            style={"textAlign": "center", "maxWidth": "800px", "margin": "0 auto 20px"},
        ),
        html.Div(
            style={
                "display": "flex",
                "gap": "20px",
                "flexWrap": "wrap",
                "alignItems": "flex-start",
                "justifyContent": "center",
            },
            children=[
                html.Div(
                    style={
                        "flex": "2 1 500px",
                        "backgroundColor": "white",
                        "borderRadius": "16px",
                        "padding": "10px 10px 0",
                        "boxShadow": "0 10px 30px rgba(0,0,0,0.05)",
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "padding": "0 12px",
                                "marginBottom": "5px",
                            },
                            children=[
                                html.Div(
                                    children=[
                                        html.Label("Center movie", style={"fontWeight": "600"}),
                                        dcc.Dropdown(
                                            id="center-movie-dropdown",
                                            options=[
                                                {
                                                    "label": row["title"],
                                                    "value": int(row["movieId"]),
                                                }
                                                for _, row in movies_vad.sample(
                                                    min(80, len(movies_vad)), random_state=0
                                                ).sort_values("title").iterrows()
                                            ],
                                            value=default_center_movie_id,
                                            clearable=False,
                                            style={"minWidth": "260px"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Label("View mode", style={"fontWeight": "600"}),
                                        dcc.RadioItems(
                                            id="mode-toggle",
                                            options=[
                                                {"label": " Simplified", "value": "simplified"},
                                                {"label": " Intense", "value": "intense"},
                                            ],
                                            value="simplified",
                                            labelStyle={"marginRight": "12px"},
                                            style={"display": "flex", "alignItems": "center"},
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="movie-network",
                            figure=build_network_figure(default_center_movie_id, "simplified"),
                            config={
                                "displayModeBar": False,
                                "scrollZoom": False,
                            },
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "flex": "1 1 260px",
                        "backgroundColor": "white",
                        "borderRadius": "16px",
                        "padding": "16px",
                        "boxShadow": "0 10px 30px rgba(0,0,0,0.05)",
                        "maxWidth": "360px",
                    },
                    children=[
                        html.H4("Movie details", style={"marginTop": 0}),
                        html.P(
                            "Hover over any node to see emotional and genre details. "
                            "Click a node to make it the new center.",
                            style={"fontSize": "0.9rem", "color": "#555"},
                        ),
                        html.Div(id="movie-details", style={"marginTop": "10px"}),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------
# 7. Dash callbacks
# ---------------------------

@app.callback(
    Output("movie-network", "figure"),
    Input("center-movie-dropdown", "value"),
    Input("mode-toggle", "value"),
)
def update_network(center_movie_id, mode):
    return build_network_figure(center_movie_id, mode)


@app.callback(
    Output("center-movie-dropdown", "value"),
    Input("movie-network", "clickData"),
    State("center-movie-dropdown", "value"),
)
def recenter_on_click(clickData, current_center):
    # When a node is clicked, use its movieId (stored in customdata) as new center
    if clickData and "points" in clickData and clickData["points"]:
        point = clickData["points"][0]
        if "customdata" in point and point["customdata"] is not None:
            return int(point["customdata"])
    return current_center


@app.callback(
    Output("movie-details", "children"),
    Input("movie-network", "hoverData"),
    State("center-movie-dropdown", "value"),
)
def update_movie_details(hoverData, center_movie_id):
    # If hovering some node, show its info, otherwise show center movie info
    movie_id = None
    if hoverData and "points" in hoverData and hoverData["points"]:
        point = hoverData["points"][0]
        if "customdata" in point:
            movie_id = int(point["customdata"])

    if movie_id is None:
        movie_id = center_movie_id

    info = movie_info(movie_id)

    return html.Div(
        children=[
            html.H5(info["title"], style={"marginBottom": "4px"}),
            html.P(f"Genres: {info['genres']}", style={"fontSize": "0.9rem"}),
            html.P(f"Mean Rating: {info['rating']:.2f}", style={"fontSize": "0.9rem"}),
            html.Hr(),
            html.P(
                f"Valence: {info['valence']:.3f} | "
                f"Arousal: {info['arousal']:.3f} | "
                f"Dominance: {info['dominance']:.3f}",
                style={"fontFamily": "monospace", "fontSize": "0.9rem"},
            ),
            html.P(
                f"Emotional Intensity: {info['intensity']:.3f}",
                style={"fontFamily": "monospace", "fontSize": "0.9rem"},
            ),
        ]
    )


if __name__ == "__main__":
 app.run(debug=True)
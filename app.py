"""FPL ML Predictor - Streamlit Dashboard."""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.predictor import FPLPredictor
from config import MODEL_NAMES, FDR_COLORS, FDR_LABELS

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FPL ML Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  CUSTOM CSS (loaded from styles.html)
# ============================================================
_styles_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.html")
with open(_styles_path, "r") as _f:
    st.markdown(_f.read(), unsafe_allow_html=True)

# ============================================================
#  SESSION STATE
# ============================================================
if "predictor" not in st.session_state:
    st.session_state.predictor = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "current_model_name" not in st.session_state:
    st.session_state.current_model_name = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# ============================================================
#  AUTO-LOAD DATA ON STARTUP
# ============================================================
if not st.session_state.data_loaded:
    with st.spinner("Loading FPL data..."):
        predictor = FPLPredictor()
        predictor.load_data()
        predictor.prepare_training_data()
        st.session_state.predictor = predictor
        st.session_state.data_loaded = True
    st.rerun()

predictor = st.session_state.predictor

# ============================================================
#  HELPER FUNCTIONS
# ============================================================
def get_status_badge(row):
    """Return HTML for injury/availability status badge."""
    status = row.get("status", "a")
    news = row.get("news", "") or ""
    chance = row.get("chance_next_round", None)

    if status == "i":
        return f'<span class="status-badge status-injured" title="{news}">INJURED</span>'
    if status == "s":
        return f'<span class="status-badge status-suspended" title="{news}">SUSPENDED</span>'
    if status == "d":
        pct = f" ({chance}%)" if chance is not None else ""
        return f'<span class="status-badge status-doubtful" title="{news}">DOUBTFUL{pct}</span>'
    if chance is not None and chance < 100 and news:
        return f'<span class="status-badge status-doubtful" title="{news}">{chance}%</span>'
    return ""


def fdr_badge_html(difficulty, opponent, is_home):
    """Generate HTML for a fixture difficulty badge."""
    venue = "H" if is_home else "A"
    label = FDR_LABELS.get(difficulty, "Medium")
    return (
        f'<span class="fdr-badge fdr-{difficulty}" title="{label}">'
        f'{opponent} ({venue})'
        f'</span>'
    )


def pos_badge_html(position):
    """Generate HTML for a position badge."""
    return f'<span class="pos-badge pos-{position}">{position}</span>'


def player_card_html(rank, row):
    """Generate HTML for a player card in the predictions table."""
    rank_class = "rank top3" if rank <= 3 else "rank"
    photo = row.get("photo_url", "")
    name = row.get("web_name", "")
    team = row.get("team", "")
    position = row.get("position", "")
    price = row.get("now_cost", 0)
    pts = row.get("predicted_points", 0)
    form = row.get("form_value", 0)
    difficulty = int(row.get("next_difficulty", 3))
    opponent = row.get("next_opponent", "???")
    is_home = row.get("next_is_home", False)
    prev_pts = row.get("prev_season_points", 0)

    fdr = fdr_badge_html(difficulty, opponent, is_home)
    pos = pos_badge_html(position)
    injury = get_status_badge(row)

    return f"""
    <div class="player-card">
        <div class="{rank_class}">{rank}</div>
        <div class="photo-container">
            <img src="{photo}" alt="{name}" onerror="this.src='https://resources.premierleague.com/premierleague/photos/players/250x250/Photo-Missing.png'">
        </div>
        <div class="info">
            <div class="name">{name}{injury}</div>
            <div class="meta">{pos} {team} · £{price:.1f}m · Form: {form:.1f}</div>
        </div>
        <div class="stats">
            <div class="stat-item">
                <div class="stat-val">{prev_pts}</div>
                <div class="stat-label">Prev Szn</div>
            </div>
            <div class="stat-item">
                {fdr}
                <div class="stat-label" style="margin-top:2px">Next Match</div>
            </div>
        </div>
        <div class="predicted-pts">
            <div class="pts-value">{pts:.1f}</div>
            <div class="pts-label">Pred Pts</div>
        </div>
    </div>
    """


def top_pick_card_html(row):
    """Generate HTML for a top pick mini card."""
    photo = row.get("photo_url", "")
    name = row.get("web_name", "")
    team = row.get("team", "")
    price = row.get("now_cost", 0)
    pts = row.get("predicted_points", 0)
    difficulty = int(row.get("next_difficulty", 3))
    opponent = row.get("next_opponent", "???")
    is_home = row.get("next_is_home", False)
    fdr = fdr_badge_html(difficulty, opponent, is_home)
    injury = get_status_badge(row)

    return f"""
    <div class="top-pick-card">
        <div class="tp-photo">
            <img src="{photo}" alt="{name}" onerror="this.src='https://resources.premierleague.com/premierleague/photos/players/250x250/Photo-Missing.png'">
        </div>
        <div class="tp-info">
            <div class="tp-name">{name}{injury}</div>
            <div class="tp-meta">{team} · £{price:.1f}m · {fdr}</div>
        </div>
        <div class="tp-pts">{pts:.1f}</div>
    </div>
    """

# ============================================================
#  SIDEBAR (simplified — model + position filter + train only)
# ============================================================
st.sidebar.markdown("""
<div style="text-align:center; padding: 1rem 0;">
    <img src="https://www.sports-betting-winners.com/wp-content/uploads/2021/12/premier-league-logo-green.png"
         style="width: 200px; height: auto; margin-bottom: 0.5rem;"
         alt="Premier League">
    <div style="font-size: 1.1rem; font-weight: 800; color: #04f5ff; letter-spacing: -0.5px;">Machine Learning Predictions</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Model selector
selected_model = st.sidebar.selectbox(
    "🤖 Select ML Model",
    options=MODEL_NAMES,
    index=0,
    help="Choose which model to use for predictions",
)

# Position filter
position_filter = st.sidebar.selectbox(
    "🏃 Filter by Position",
    options=["All", "GK", "DEF", "MID", "FWD"],
    index=0,
)

# Top N slider
top_n = st.sidebar.slider(
    "📊 Show Top N Players",
    min_value=10,
    max_value=100,
    value=30,
    step=5,
)

st.sidebar.markdown("---")

# Train model button
if st.sidebar.button("🚀 Train Model", type="primary", use_container_width=True):
    with st.spinner(f"Training {selected_model}..."):
        metrics = predictor.train_model(selected_model)
        predictions = predictor.predict()
        st.session_state.predictions = predictions
        st.session_state.metrics = metrics
        st.session_state.model_trained = True
        st.session_state.current_model_name = selected_model
    st.sidebar.success(f"{selected_model} trained successfully!")
    st.rerun()

# Data info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Data Info")
st.sidebar.write(f"**Current GW:** {predictor.current_gw}")
st.sidebar.write(f"**Next GW:** {predictor.next_gw}")
st.sidebar.write(f"**Players:** {len(predictor.all_histories)}")
st.sidebar.write(f"**Training rows:** {len(predictor.training_df):,}")

# FDR Legend
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Difficulty Legend")
legend_html = ""
for fdr_val in [1, 2, 3, 4, 5]:
    legend_html += (
        f'<span class="fdr-badge fdr-{fdr_val}" style="margin-right:4px; margin-bottom:4px;">'
        f'FDR {fdr_val} - {FDR_LABELS[fdr_val]}'
        f'</span> '
    )
st.sidebar.markdown(f'<div class="fdr-row">{legend_html}</div>', unsafe_allow_html=True)

# ============================================================
#  MAIN CONTENT
# ============================================================

# Header
st.markdown(f"""
<div class="fpl-header">
    <h1>⚽ Fantasy Premier League ML Predictor</h1>
    <p>Predict player points for Gameweek {predictor.next_gw} using machine learning</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.model_trained:
    st.divider()
    st.markdown("""
    <div style="text-align: center;">
        <h3>👈 <strong>Select a model</strong> and click <strong>Train Model</strong> in the sidebar to generate predictions.</h3>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.stop()

# ============================================================
#  MODEL RESULTS
# ============================================================
metrics = st.session_state.metrics

# --- Metrics Row ---
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="label">Model</div>
        <div class="value purple">{st.session_state.current_model_name}</div>
    </div>
    <div class="metric-card">
        <div class="label">MAE</div>
        <div class="value">{metrics['MAE']:.2f}</div>
    </div>
    <div class="metric-card">
        <div class="label">RMSE</div>
        <div class="value">{metrics['RMSE']:.2f}</div>
    </div>
    <div class="metric-card">
        <div class="label">R²</div>
        <div class="value green">{metrics['R2']:.3f}</div>
    </div>
    <div class="metric-card">
        <div class="label">Next Gameweek</div>
        <div class="value yellow">GW {predictor.next_gw}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
#  TABS (centered)
# ============================================================
tab_predictions, tab_players, tab_analysis = st.tabs([
    "🎯 Predictions", "🔍 Player Search", "📊 Model Analysis"
])

# ============================================================
#  TAB 1: PREDICTIONS (Top 5 per position + ranked list)
# ============================================================
with tab_predictions:

    # --- Top 5 Picks by Position ---
    st.markdown(f'<div class="section-header"><h2>🏆 Top 5 Picks by Position — GW {predictor.next_gw}</h2></div>', unsafe_allow_html=True)

    all_predictions = st.session_state.predictions
    pos_cols = st.columns(4)

    for i, pos in enumerate(["GK", "DEF", "MID", "FWD"]):
        with pos_cols[i]:
            st.markdown(f"{pos_badge_html(pos)}", unsafe_allow_html=True)
            pos_picks = all_predictions[all_predictions["position"] == pos].head(5)
            picks_html = ""
            for _, row in pos_picks.iterrows():
                picks_html += top_pick_card_html(row)
            st.markdown(picks_html, unsafe_allow_html=True)

    # --- Full Ranked List ---
    st.markdown(
        f'<div class="section-header"><h2>🎯 Full Rankings — Gameweek {predictor.next_gw}</h2></div>',
        unsafe_allow_html=True
    )

    predictions = st.session_state.predictions.copy()

    # Apply position filter from sidebar
    if position_filter != "All":
        predictions = predictions[predictions["position"] == position_filter]

    display_df = predictions.head(top_n)

    if len(display_df) == 0:
        st.warning("No players match your filters.")
    else:
        cards_html = ""
        for i, (_, row) in enumerate(display_df.iterrows(), 1):
            cards_html += player_card_html(i, row)
        st.markdown(cards_html, unsafe_allow_html=True)

# ============================================================
#  TAB 2: PLAYER SEARCH (in-tab search + results)
# ============================================================
with tab_players:
    st.markdown('<div class="section-header"><h2>🔍 Player Search</h2></div>', unsafe_allow_html=True)

    # In-tab search controls
    search_col1, search_col2, search_col3 = st.columns([3, 1, 1])
    with search_col1:
        player_search = st.text_input(
            "Search by name",
            placeholder="e.g. Salah, Haaland, Palmer...",
            key="tab_player_search",
        )
    with search_col2:
        search_pos = st.selectbox(
            "Position",
            options=["All", "GK", "DEF", "MID", "FWD"],
            key="tab_search_pos",
        )
    with search_col3:
        show_injured = st.checkbox("Include doubtful", value=True, key="tab_show_injured")

    search_results = st.session_state.predictions.copy()

    # Filter by position
    if search_pos != "All":
        search_results = search_results[search_results["position"] == search_pos]

    # Filter by search query
    if player_search:
        search_results = search_results[
            search_results["web_name"].str.contains(player_search, case=False, na=False)
            | search_results["first_name"].str.contains(player_search, case=False, na=False)
            | search_results["second_name"].str.contains(player_search, case=False, na=False)
        ]

    # Optionally exclude doubtful players
    if not show_injured:
        search_results = search_results[
            search_results["status"].isin(["a", ""])
            | search_results["status"].isna()
        ]

    st.caption(f"Showing {len(search_results)} player(s)")

    if len(search_results) == 0:
        st.info("No players found. Try a different search term.")
    else:
        cards_html = ""
        for i, (_, row) in enumerate(search_results.head(50).iterrows(), 1):
            cards_html += player_card_html(i, row)
        st.markdown(cards_html, unsafe_allow_html=True)

# ============================================================
#  TAB 3: MODEL ANALYSIS
# ============================================================
with tab_analysis:
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown('<div class="section-header"><h2>📦 Points Distribution by Position</h2></div>', unsafe_allow_html=True)
        fig_box = px.box(
            st.session_state.predictions,
            x="position",
            y="predicted_points",
            color="position",
            color_discrete_map={"GK": "#f59e0b", "DEF": "#10b981", "MID": "#3b82f6", "FWD": "#ef4444"},
            labels={"predicted_points": "Predicted Points", "position": "Position"},
            category_orders={"position": ["GK", "DEF", "MID", "FWD"]},
        )
        fig_box.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c8c8d8",
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with viz_col2:
        st.markdown('<div class="section-header"><h2>🔬 Feature Importance</h2></div>', unsafe_allow_html=True)
        fi_df = predictor.get_feature_importance()
        fig_fi = px.bar(
            fi_df.head(15),
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Importance", "feature": "Feature"},
            color="importance",
            color_continuous_scale=["#37003c", "#04f5ff"],
        )
        fig_fi.update_layout(
            yaxis=dict(autorange="reversed"),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c8c8d8",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Actual vs Predicted
    st.markdown('<div class="section-header"><h2>📈 Model Validation: Actual vs Predicted</h2></div>', unsafe_allow_html=True)

    avp_df = predictor.get_actual_vs_predicted()

    val_col1, val_col2 = st.columns(2)

    with val_col1:
        gw_agg = (
            avp_df.groupby("gameweek")
            .agg(actual_mean=("actual", "mean"), predicted_mean=("predicted", "mean"))
            .reset_index()
        )

        fig_avp = go.Figure()
        fig_avp.add_trace(
            go.Scatter(
                x=gw_agg["gameweek"],
                y=gw_agg["actual_mean"],
                mode="lines+markers",
                name="Actual (mean)",
                line=dict(color="#04f5ff", width=2),
                marker=dict(size=6),
            )
        )
        fig_avp.add_trace(
            go.Scatter(
                x=gw_agg["gameweek"],
                y=gw_agg["predicted_mean"],
                mode="lines+markers",
                name="Predicted (mean)",
                line=dict(color="#00ff87", dash="dash", width=2),
                marker=dict(size=6),
            )
        )
        fig_avp.update_layout(
            title="Mean Points per Gameweek",
            xaxis_title="Gameweek",
            yaxis_title="Mean Points",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c8c8d8",
        )
        st.plotly_chart(fig_avp, use_container_width=True)

    with val_col2:
        fig_scatter = px.scatter(
            avp_df,
            x="actual",
            y="predicted",
            hover_data=["web_name", "gameweek"],
            title="Predicted vs Actual (each dot = player-gameweek)",
            labels={"actual": "Actual Points", "predicted": "Predicted Points"},
            opacity=0.3,
        )
        fig_scatter.update_traces(marker=dict(color="#04f5ff"))
        max_val = max(
            avp_df["actual"].max() if len(avp_df) > 0 else 15,
            avp_df["predicted"].max() if len(avp_df) > 0 else 15,
        )
        fig_scatter.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="#00ff87", dash="dash"),
            )
        )
        fig_scatter.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c8c8d8",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Fixture Difficulty Overview for next GW
    st.markdown(f'<div class="section-header"><h2>🗓️ Fixture Difficulty — GW {predictor.next_gw}</h2></div>', unsafe_allow_html=True)

    fixture_data = st.session_state.predictions[
        ["team", "next_opponent", "next_difficulty", "next_is_home"]
    ].drop_duplicates(subset=["team"]).sort_values("next_difficulty")

    fdr_cards = '<div class="fdr-row">'
    for _, frow in fixture_data.iterrows():
        diff = int(frow["next_difficulty"])
        opp = frow["next_opponent"]
        venue = "H" if frow["next_is_home"] else "A"
        team = frow["team"]
        fdr_cards += (
            f'<span class="fdr-badge fdr-{diff}" style="padding:6px 12px; font-size:0.8rem;">'
            f'{team} vs {opp} ({venue})'
            f'</span>'
        )
    fdr_cards += '</div>'
    st.markdown(fdr_cards, unsafe_allow_html=True)

    # ===========================================================
    #  TOP 5 STATISTICAL DEEP DIVE
    # ===========================================================
    st.markdown(f'<div class="section-header"><h2>🔬 Top 5 Deep Dive — Why the Model Rates Them Highest</h2></div>', unsafe_allow_html=True)

    # Pull top 5 from the full prediction_df (has all ML feature columns)
    pred_full = predictor.prediction_df.copy()
    top5_ids = st.session_state.predictions.head(5)["player_id"].tolist()
    top5_full = pred_full[pred_full["player_id"].isin(top5_ids)].copy()
    # Merge predicted points
    top5_full = top5_full.merge(
        st.session_state.predictions[["player_id", "predicted_points"]],
        on="player_id",
    )
    top5_full = top5_full.sort_values("predicted_points", ascending=False)

    # Get feature importance to know the top drivers
    fi_dict = dict(zip(fi_df["feature"], fi_df["importance"]))

    # All-player averages for comparison
    all_avg = {}
    stat_cols = [
        "rolling_3gw_points", "rolling_5gw_points", "rolling_3gw_goals",
        "rolling_3gw_assists", "rolling_3gw_bonus", "rolling_5gw_goals",
        "rolling_5gw_assists", "influence", "creativity", "threat",
        "ict_index", "goals_per_90", "assists_per_90", "selected_by_percent",
        "clean_sheet_probability", "xg_overperformance", "minutes_trend",
    ]
    for c in stat_cols:
        all_avg[c] = float(pred_full[c].mean()) if c in pred_full.columns else 0.0

    for idx, (_, p) in enumerate(top5_full.iterrows(), 1):
        name = p.get("web_name", "")
        photo = p.get("photo_url", "")
        team = p.get("team", "")
        pos = p.get("position", "")
        price = p.get("now_cost", 0)
        pts = p.get("predicted_points", 0)
        diff = int(p.get("next_difficulty", 3))
        opp = p.get("next_opponent", "???")
        is_home = p.get("next_is_home", False)
        venue_text = "at home" if is_home else "away"
        fdr_label = FDR_LABELS.get(diff, "Medium")
        injury = get_status_badge(p)
        prev_pts = p.get("prev_season_points", 0)
        prev_szn = p.get("prev_season_name", "N/A")

        # Key stats
        r3_pts = p.get("rolling_3gw_points", 0)
        r5_pts = p.get("rolling_5gw_points", 0)
        r3_g = p.get("rolling_3gw_goals", 0)
        r3_a = p.get("rolling_3gw_assists", 0)
        r3_bonus = p.get("rolling_3gw_bonus", 0)
        infl = p.get("influence", 0)
        creat = p.get("creativity", 0)
        thr = p.get("threat", 0)
        ict = p.get("ict_index", 0)
        gp90 = p.get("goals_per_90", 0)
        ap90 = p.get("assists_per_90", 0)
        own_pct = p.get("selected_by_percent", 0)
        cs_prob = p.get("clean_sheet_probability", 0)
        xg_over = p.get("xg_overperformance", 0)
        min_trend = p.get("minutes_trend", 0)
        form = p.get("form_value", 0)

        # Build "why" bullet points based on strongest signals
        reasons = []

        if r3_pts > all_avg.get("rolling_3gw_points", 0) * 1.3:
            reasons.append(f"Averaging **{r3_pts:.1f} pts** over last 3 GWs (league avg: {all_avg['rolling_3gw_points']:.1f})")
        elif r3_pts > 0:
            reasons.append(f"Averaging **{r3_pts:.1f} pts** over last 3 GWs")

        if r3_g > 0 or r3_a > 0:
            parts = []
            if r3_g > 0:
                parts.append(f"{int(r3_g)} goal{'s' if r3_g > 1 else ''}")
            if r3_a > 0:
                parts.append(f"{int(r3_a)} assist{'s' if r3_a > 1 else ''}")
            reasons.append(f"**{' + '.join(parts)}** in last 3 GWs")

        if r3_bonus > all_avg.get("rolling_3gw_bonus", 0) * 1.5 and r3_bonus > 1:
            reasons.append(f"Earning **{int(r3_bonus)} bonus points** recently (high BPS)")

        if diff <= 2:
            reasons.append(f"**{fdr_label} fixture** {venue_text} vs {opp} (FDR {diff})")
        elif diff >= 4:
            reasons.append(f"Tough fixture {venue_text} vs {opp} (FDR {diff}) — but form overrides")
        else:
            reasons.append(f"Balanced fixture {venue_text} vs {opp} (FDR {diff})")

        if ict > all_avg.get("ict_index", 0) * 1.5:
            reasons.append(f"ICT Index **{ict:.1f}** — elite influence ({infl:.0f}), creativity ({creat:.0f}), threat ({thr:.0f})")

        if gp90 > 0.3:
            reasons.append(f"Goal threat: **{gp90:.2f} goals/90**")
        if ap90 > 0.3:
            reasons.append(f"Creative output: **{ap90:.2f} assists/90**")

        if pos in ("GK", "DEF") and cs_prob > 0.3:
            reasons.append(f"**{cs_prob:.0%}** clean sheet probability")

        if xg_over > 1:
            reasons.append(f"Outperforming xG by **{xg_over:.1f}** goals — clinical finishing")
        elif xg_over < -1:
            reasons.append(f"Underperforming xG by **{abs(xg_over):.1f}** — due a correction upwards")

        if prev_pts > 150:
            reasons.append(f"**{prev_pts} pts** last season ({prev_szn}) — proven pedigree")

        if own_pct > 30:
            reasons.append(f"Owned by **{own_pct:.1f}%** of managers — essential pick")

        if min_trend > 5:
            reasons.append(f"Minutes trending **upward** — gaining more game time")

        # Render the card
        fdr = fdr_badge_html(diff, opp, is_home)
        reasons_html = "".join(f"<li>{r}</li>" for r in reasons[:6])

        st.markdown(f"""
        <div style="background:linear-gradient(145deg,#1a1a2e 0%,#0f0f23 100%); border-radius:12px; padding:1.2rem; margin-bottom:1rem; border:1px solid #2a2a4a;">
            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.8rem;">
                <div style="font-size:1.6rem; font-weight:800; color:#00ff87; min-width:30px;">#{idx}</div>
                <div style="width:52px; height:52px; border-radius:50%; overflow:hidden; border:2px solid #2a2a4a; flex-shrink:0; background:#16213e;">
                    <img src="{photo}" style="width:100%; height:100%; object-fit:cover;"
                         onerror="this.src='https://resources.premierleague.com/premierleague/photos/players/250x250/Photo-Missing.png'">
                </div>
                <div style="flex:1;">
                    <div style="font-weight:700; font-size:1.1rem; color:#e6e6e6;">{name}{injury}</div>
                    <div style="font-size:0.8rem; color:#8892b0;">
                        {pos_badge_html(pos)} {team} · £{price:.1f}m · Form: {form:.1f} · {fdr}
                    </div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:2rem; font-weight:800; color:#00ff87;">{pts:.1f}</div>
                    <div style="font-size:0.65rem; color:#8892b0; text-transform:uppercase;">Predicted</div>
                </div>
            </div>
            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:0.5rem; margin-bottom:0.8rem;">
                <div style="background:#16213e; border-radius:8px; padding:0.5rem; text-align:center;">
                    <div style="font-size:0.6rem; color:#8892b0; text-transform:uppercase;">3GW Avg</div>
                    <div style="font-size:1.1rem; font-weight:700; color:#04f5ff;">{r3_pts:.1f}</div>
                </div>
                <div style="background:#16213e; border-radius:8px; padding:0.5rem; text-align:center;">
                    <div style="font-size:0.6rem; color:#8892b0; text-transform:uppercase;">5GW Avg</div>
                    <div style="font-size:1.1rem; font-weight:700; color:#04f5ff;">{r5_pts:.1f}</div>
                </div>
                <div style="background:#16213e; border-radius:8px; padding:0.5rem; text-align:center;">
                    <div style="font-size:0.6rem; color:#8892b0; text-transform:uppercase;">ICT</div>
                    <div style="font-size:1.1rem; font-weight:700; color:#04f5ff;">{ict:.1f}</div>
                </div>
                <div style="background:#16213e; border-radius:8px; padding:0.5rem; text-align:center;">
                    <div style="font-size:0.6rem; color:#8892b0; text-transform:uppercase;">Prev Szn</div>
                    <div style="font-size:1.1rem; font-weight:700; color:#04f5ff;">{prev_pts}</div>
                </div>
            </div>
            <div style="font-size:0.85rem; color:#c8c8d8;">
                <strong style="color:#04f5ff;">Why the model rates {name} highly:</strong>
                <ul style="margin:0.3rem 0 0; padding-left:1.2rem; line-height:1.6;">{reasons_html}</ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Radar chart comparing top 5 across key dimensions
    st.markdown('<div class="section-header"><h2>📊 Top 5 Stat Comparison</h2></div>', unsafe_allow_html=True)

    radar_categories = ["Form (3GW)", "Form (5GW)", "ICT Index", "Goals/90", "Assists/90", "Ownership %"]

    fig_radar = go.Figure()
    for _, p in top5_full.iterrows():
        # Normalize each stat 0-100 for radar display
        max_r3 = pred_full["rolling_3gw_points"].max() or 1
        max_r5 = pred_full["rolling_5gw_points"].max() or 1
        max_ict = pred_full["ict_index"].max() or 1
        max_gp90 = pred_full["goals_per_90"].max() or 1
        max_ap90 = pred_full["assists_per_90"].max() or 1
        max_own = pred_full["selected_by_percent"].max() or 1

        values = [
            (p.get("rolling_3gw_points", 0) / max_r3) * 100,
            (p.get("rolling_5gw_points", 0) / max_r5) * 100,
            (p.get("ict_index", 0) / max_ict) * 100,
            (p.get("goals_per_90", 0) / max_gp90) * 100 if max_gp90 > 0 else 0,
            (p.get("assists_per_90", 0) / max_ap90) * 100 if max_ap90 > 0 else 0,
            (p.get("selected_by_percent", 0) / max_own) * 100,
        ]
        values.append(values[0])  # close the polygon

        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_categories + [radar_categories[0]],
            fill="toself",
            name=f"{p['web_name']} ({p['predicted_points']:.1f}pts)",
            opacity=0.6,
        ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="#2a2a4a"),
            angularaxis=dict(gridcolor="#2a2a4a", linecolor="#2a2a4a"),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#c8c8d8",
        height=500,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

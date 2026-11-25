# -*- coding: utf-8 -*-
"""Customer Complaints Analytics.

Adapted from Plotly's open-source `dash-nlp` sample app (MIT licence, see LICENSE; the original was
written for Plotly by Vildly in 2019). Rebuilt and extended by Pawel Sogra:

* rebuilt for current Dash 4 / dash-bootstrap-components 2 / pandas 3 (the 2020 stack, including
  spaCy 2 and gensim 3, no longer installs on current Python);
* runtime no longer needs spaCy, gensim or matplotlib, so it starts fast and fits small hosts;
* new complaint-volume trend and a transparent keyword-based theme tracker;
* month-range slider replaces the epoch-timestamp slider; stop-words no longer leak between
  requests; heavy computations are cached.
"""
import io
import json
import os
import pathlib
from functools import lru_cache

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html
from sklearn.manifold import TSNE
from wordcloud import STOPWORDS, WordCloud

DATA_PATH = pathlib.Path(__file__).parent.resolve() / "data"
PORTFOLIO_URL = os.environ.get("PORTFOLIO_URL", "")
DEFAULT_BANK = "EQUIFAX, INC."

# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
GLOBAL_DF = pd.read_csv(DATA_PATH / "customer_complaints_narrative_sample.csv.gz", index_col=0)
GLOBAL_DF["Date received"] = pd.to_datetime(GLOBAL_DF["Date received"], format="%m/%d/%Y")

embed_df = pd.read_csv(DATA_PATH / "tsne_bigram_data.csv", index_col=0)
vects_df = pd.read_csv(DATA_PATH / "bigram_vectors.csv", index_col=0)  # averaged GloVe 50d vectors
bigram_df = pd.read_csv(DATA_PATH / "bigram_counts_data.csv", index_col=0)

# Precomputed offline: per company, LDA (5 topics) on spaCy-lemmatised text + t-SNE layout.
with open(DATA_PATH / "precomputed.json") as fh:
    PRECOMPUTED_LDA = json.load(fh)

MONTHS = pd.date_range(
    GLOBAL_DF["Date received"].min().replace(day=1),
    GLOBAL_DF["Date received"].max().replace(day=1),
    freq="MS",
)
N_MONTHS = len(MONTHS)
GLOBAL_DF["month_idx"] = (
    (GLOBAL_DF["Date received"].dt.year - MONTHS[0].year) * 12
    + GLOBAL_DF["Date received"].dt.month
    - MONTHS[0].month
)

# Keyword-based themes built from the dataset's bigram flags. A complaint can have several themes.
THEMES = {
    "Credit reporting": [
        "credit_report", "credit_reporting", "credit_bureaus", "credit_bureau", "credit_score",
        "reporting_agency", "report_credit", "remove_credit", "credit_file", "information_credit",
    ],
    "Debt collection": [
        "collection_agency", "debt_collection", "debt_collector", "collect_debt", "past_due",
        "receive_call",
    ],
    "Payments and fees": [
        "make_payment", "payment_make", "late_payment", "late_fee", "monthly_payment", "pay_full",
        "interest_rate", "day_late",
    ],
    "Loans and mortgages": [
        "student_loan", "loan_modification", "mortgage_payment", "mortgage_company", "short_sale",
    ],
    "Customer service": ["customer_service", "call_back", "phone_call", "phone_number", "please_help"],
    "Fraud and identity": ["identity_theft", "social_security"],
}
for _theme, _cols in THEMES.items():
    GLOBAL_DF["theme: " + _theme] = GLOBAL_DF[_cols].any(axis=1)
THEME_COLS = ["theme: " + t for t in THEMES]

ADDITIONAL_STOPWORDS = {"xxxx", "xx", "n't", "'s", "s", "t", "account"}
BASE_STOPWORDS = {w.lower() for w in STOPWORDS} | ADDITIONAL_STOPWORDS

_counts = GLOBAL_DF["Company"].value_counts()
BANK_OPTIONS = [{"label": b, "value": b} for b in _counts[_counts > 10].index]
COMPARE_COMPANIES = sorted(bigram_df.company.unique())


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------
def month_label(i):
    return MONTHS[int(i)].strftime("%b %Y")


def in_range(df, m0, m1):
    return df[(df["month_idx"] >= m0) & (df["month_idx"] <= m1)]


def select(bank, m0, m1):
    d = in_range(GLOBAL_DF, m0, m1)
    return d[d["Company"] == bank] if bank else d


def make_marks():
    marks = {}
    for i, m in enumerate(MONTHS):
        if m.month in (1, 7):
            marks[i] = {"label": m.strftime("%b %y"), "style": {"fontSize": 10}}
    return marks


def empty_fig(message="No data for this selection"):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color="#667"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template="plotly_white", margin=dict(t=20, b=20, l=20, r=20))
    return fig


@lru_cache(maxsize=64)
def word_figures(bank, m0, m1):
    """Word frequency bar, treemap and wordcloud for a bank and month range (cached)."""
    d = select(bank, m0, m1)
    texts = d["Consumer complaint narrative"].dropna().tolist()
    if not texts:
        return None
    stop = set(BASE_STOPWORDS)
    if bank:  # drop the company's own name words for this request only
        stop |= {w.lower() for w in "".join(c if c.isalnum() else " " for c in bank).split()}

    wc = WordCloud(stopwords=stop, max_words=100, max_font_size=90, random_state=1)
    wc.generate(" ".join(texts))
    words, freqs, positions, colors = [], [], [], []
    for (word, freq), _size, position, _orient, color in wc.layout_:
        words.append(word)
        freqs.append(freq)
        positions.append(position)
        colors.append(color)

    cloud = go.Figure(
        go.Scatter(
            x=[p[0] for p in positions],
            y=[p[1] for p in positions],
            mode="text",
            text=words,
            textposition="top center",
            textfont=dict(size=[f * 80 for f in freqs], color=colors),
            hoverinfo="text",
            hovertext=[f"{w}: {f:.2f} (relative)" for w, f in zip(words, freqs)],
        )
    )
    cloud.update_layout(
        template="plotly_white",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-100, 250]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-100, 450]),
        margin=dict(t=20, b=20, l=10, r=10),
        hovermode="closest",
        height=500,
    )

    top_words, top_freqs = words[:25][::-1], freqs[:25][::-1]
    bars = go.Figure(go.Bar(y=top_words, x=top_freqs, orientation="h", marker_color="#2c82ff"))
    bars.update_layout(
        template="plotly_white", height=550, margin=dict(t=20, b=30, l=100, r=20),
        xaxis_title="Relative frequency",
    )
    tree = go.Figure(go.Treemap(labels=top_words, parents=[""] * len(top_words), values=top_freqs))
    tree.update_layout(margin=dict(t=10, b=10, l=5, r=5), height=550)
    return bars, tree, cloud


@lru_cache(maxsize=16)
def lda_for_bank(bank):
    """Scatter figure and table rows from the precomputed LDA output for one company."""
    entry = PRECOMPUTED_LDA.get(bank)
    if entry is None:
        return None
    dominant = pd.read_json(io.StringIO(entry["df_dominant_topic"]))
    tsne = pd.read_json(io.StringIO(entry["tsne_df"]))
    top3 = pd.read_json(io.StringIO(entry["df_top3words"]))
    if "Date" in dominant.columns:
        dominant["Date"] = pd.to_datetime(dominant["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    palette = px.colors.qualitative.Plotly
    fig = go.Figure()
    for topic_id in top3["topic_id"]:
        sub = tsne[tsne["topic_num"] == topic_id]
        words = top3.loc[top3["topic_id"] == topic_id, "words"].iloc[0].replace("\n", "")
        fig.add_trace(
            go.Scatter(
                name=words, x=sub["tsne_x"], y=sub["tsne_y"], mode="markers",
                hovertext=sub["doc_num"],
                hovertemplate="Complaint #%{hovertext}<extra>" + words + "</extra>",
                marker=dict(size=6, color=palette[int(topic_id) % len(palette)]),
            )
        )
    fig.update_layout(
        template="plotly_white", title="Complaints grouped by LDA topic (t-SNE layout)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        legend=dict(orientation="h", y=-0.05), margin=dict(t=50, b=10),
    )
    columns = [{"name": c.replace("_", " "), "id": c} for c in dominant.columns]
    return fig, columns, dominant.to_dict("records")


@lru_cache(maxsize=8)
def bigram_embedding(perplexity):
    xy = TSNE(n_components=2, perplexity=perplexity, random_state=0).fit_transform(vects_df)
    out = embed_df.copy()
    out["tsne_1"], out["tsne_2"] = xy[:, 0], xy[:, 1]
    return out


# ----------------------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------------------
def section_card(title, body, subtitle=None, **kw):
    header = [html.H5(title, className="mb-0")]
    if subtitle:
        header.append(html.Small(subtitle, className="text-muted"))
    return dbc.Card([dbc.CardHeader(header), dbc.CardBody(body)], className="mb-4", **kw)


NAVBAR = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.Span("Customer Complaints Analytics", className="navbar-brand mb-0 h1"),
                    html.Span("Pawel Sogra | NLP live demo", className="nav-sub"),
                ]
            ),
            html.A(
                "Back to portfolio", href=PORTFOLIO_URL or "#", className="btn btn-outline-light btn-sm",
                style={} if PORTFOLIO_URL else {"display": "none"},
            ),
        ],
        fluid=True,
        className="d-flex justify-content-between align-items-center",
    ),
    color="dark", dark=True, sticky="top",
)

INTRO = dbc.Alert(
    [
        html.Strong("What am I looking at? "),
        f"{len(GLOBAL_DF):,} consumer complaints about US financial companies "
        f"({month_label(0)} to {month_label(N_MONTHS - 1)}), in the format of the CFPB public complaint "
        "database. Pick a company and a time frame to see volumes, themes, frequent words and topics. "
        "Narratives are consumer-written; names and amounts are redacted (XXXX).",
    ],
    color="light", className="mt-3 mb-0",
)

CONTROLS = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Select company and time frame"),
            html.Hr(className="my-2"),
            html.Label("Company", className="lead"),
            html.P(
                "Use the dropdown or click a bar in the Top 10 chart.",
                className="small text-muted mb-1",
            ),
            dcc.Dropdown(
                id="bank-drop", options=BANK_OPTIONS, value=DEFAULT_BANK, clearable=False,
                style={"fontSize": 12},
            ),
            html.Label("Time frame", className="lead mt-4"),
            html.P("Month granularity.", className="small text-muted mb-3"),
            dcc.RangeSlider(
                id="time-window-slider", min=0, max=N_MONTHS - 1, step=1, value=[0, N_MONTHS - 1],
                marks=make_marks(), allowCross=False, tooltip={"always_visible": False},
            ),
            html.Div(id="time-window-label", className="small mt-3"),
        ]
    ),
    className="mb-4",
)

LDA_BLOCK = [
    html.P(
        "Topics were precomputed offline (spaCy lemmatisation, gensim LDA with 5 topics, t-SNE layout) on "
        "all of a company's complaints, so this section ignores the time-frame slider. Topics overlap "
        "heavily, so treat them as exploratory. Click a point to read that complaint.",
        className="small text-muted",
    ),
    dbc.Alert(
        "No precomputed topic model for this company.", id="no-data-alert-lda", color="warning",
        style={"display": "none"},
    ),
    dcc.Loading(dcc.Graph(id="tsne-lda")),
    html.Hr(),
    html.Div(
        id="lda-table-block",
        style={"display": "none"},
        children=dash_table.DataTable(
            id="lda-table",
            style_cell_conditional=[
                {"if": {"column_id": "Text"}, "textAlign": "left", "minWidth": "50%"}
            ],
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "rgb(243, 246, 251)"}],
            style_cell={"padding": "12px", "whiteSpace": "normal", "height": "auto", "maxWidth": "0"},
            style_header={"backgroundColor": "white", "fontWeight": "bold"},
            filter_action="native", page_action="native", page_current=0, page_size=5,
            columns=[], data=[],
        ),
    ),
]

BODY = dbc.Container(
    [
        INTRO,
        dbc.Row(
            [
                dbc.Col(CONTROLS, md=4),
                dbc.Col(
                    [
                        html.Div(id="kpi-row", className="mb-3"),
                        dbc.Card(
                            [
                                dbc.CardHeader(html.H5("Top 10 companies by complaints (selected time frame)", className="mb-0")),
                                dbc.CardBody(dcc.Graph(id="bank-sample", config={"displayModeBar": False})),
                            ],
                            className="mb-4",
                        ),
                    ],
                    md=8,
                ),
            ],
            className="mt-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    section_card(
                        "Complaint volume over time",
                        dcc.Loading(dcc.Graph(id="trend-fig", config={"displayModeBar": False})),
                        "Bars: selected company (highlighted = selected time frame). Line: all companies.",
                    ),
                    md=6,
                ),
                dbc.Col(
                    section_card(
                        "What do complaints mention?",
                        dcc.Loading(dcc.Graph(id="theme-fig", config={"displayModeBar": False})),
                        "Keyword-based themes, not a trained classifier. A complaint can have several.",
                    ),
                    md=6,
                ),
            ]
        ),
        section_card(
            "Most frequently used words",
            [
                dbc.Alert(
                    "Not enough data to render these plots, please adjust the filters",
                    id="no-data-alert", color="warning", style={"display": "none"},
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Loading(dcc.Graph(id="frequency_figure")), md=4),
                        dbc.Col(
                            dcc.Tabs(
                                children=[
                                    dcc.Tab(label="Treemap", children=dcc.Loading(dcc.Graph(id="bank-treemap"))),
                                    dcc.Tab(label="Wordcloud", children=dcc.Loading(dcc.Graph(id="bank-wordcloud"))),
                                ]
                            ),
                            md=8,
                        ),
                    ]
                ),
            ],
        ),
        section_card("Topic modelling (LDA)", LDA_BLOCK),
        section_card(
            "Compare two companies: most used bigrams",
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(id="bigrams-comp_1", options=COMPARE_COMPANIES, value="EQUIFAX, INC.", clearable=False),
                            md=6,
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="bigrams-comp_2", options=COMPARE_COMPANIES,
                                value="TRANSUNION INTERMEDIATE HOLDINGS, INC.", clearable=False,
                            ),
                            md=6,
                        ),
                    ]
                ),
                dcc.Graph(id="bigrams-comps"),
            ],
            "Counts of the 50 most common word pairs; the second company is mirrored below the axis.",
        ),
        section_card(
            "Bigram similarity and frequency",
            [
                dbc.Row(
                    [
                        dbc.Col(html.P("t-SNE perplexity:"), width="auto"),
                        dbc.Col(
                            dcc.Dropdown(
                                id="bigrams-perplex-dropdown",
                                options=[{"label": str(i), "value": i} for i in range(3, 7)],
                                value=3, clearable=False, style={"minWidth": 80},
                            ),
                            width="auto",
                        ),
                    ]
                ),
                dcc.Loading(dcc.Graph(id="bigrams-scatter")),
            ],
            "Bigrams placed by averaged GloVe 50-d word vectors; closer means more similar in meaning.",
        ),
    ],
    fluid=True,
    className="pb-4",
)

FOOTER = html.Div(
    [
        "Demonstration data bundled with Plotly's sample app (Mar 2015 to Nov 2017); illustrative, not current. "
        "Adapted from Plotly's open-source ",
        html.A("dash-nlp", href="https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-nlp", target="_blank"),
        " (MIT licence). Rebuilt and extended by Pawel Sogra: current Dash/pandas, volume trend, theme tracker, "
        "month-range slider, caching, fixes.",
    ],
    id="footer",
)

def showcase_note():
    """Banner making clear this is a showcase on demonstration data, not a real client project."""
    link = html.A(
        "Discuss a similar project \u2192",
        href=(PORTFOLIO_URL.rstrip("/") + "/#contact") if PORTFOLIO_URL else "#",
        target="_blank",
        style={"marginLeft": "10px", "fontWeight": 600, "color": "#b0432d", "whiteSpace": "nowrap"},
    )
    return html.Div(
        id="showcase-note",
        children=[
            html.Strong("Showcase project. "),
            "Built to show what is possible, using demonstration data. This is not a real client project. "
            "The same approach works on your own data. ",
            link if PORTFOLIO_URL else "",
        ],
        style={
            "background": "#fff7ed", "color": "#7a4a12", "borderBottom": "1px solid #f3d9b8",
            "padding": "10px 24px", "fontSize": "14px", "lineHeight": "1.5", "textAlign": "center",
            "clear": "both",
        },
    )


app = dash.Dash(__name__)  # Bootstrap 5.3.3 (MIT) is bundled in assets/
app.title = "Customer Complaints Analytics | Pawel Sogra"
server = app.server
app.layout = html.Div([NAVBAR, showcase_note(), BODY, FOOTER])


# ----------------------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------------------
@app.callback(
    [Output("bank-sample", "figure"), Output("time-window-label", "children")],
    Input("time-window-slider", "value"),
)
def update_top_banks(months):
    m0, m1 = months
    d = in_range(GLOBAL_DF, m0, m1)
    top = d["Company"].value_counts().head(10)
    fig = go.Figure(
        go.Bar(
            x=top.index, y=top.values, text=top.index, textposition="auto",
            marker_color="#2c82ff", hovertemplate="%{x}: %{y} complaints<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white", margin=dict(t=10, b=10, l=40, r=0),
        xaxis=dict(showticklabels=False), height=330,
    )
    label = f"{month_label(m0)} to {month_label(m1)}: {len(d):,} complaints"
    return fig, label


@app.callback(Output("bank-drop", "value"), Input("bank-sample", "clickData"), prevent_initial_call=True)
def update_bank_drop_on_click(click):
    if click:
        return click["points"][0]["x"]
    return dash.no_update


def kpi(label, value, sub=""):
    return dbc.Col(
        html.Div(
            [html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value"),
             html.Div(sub, className="kpi-sub")],
            className="kpi-tile",
        ),
        md=3, xs=6,
    )


@app.callback(
    [Output("kpi-row", "children"), Output("trend-fig", "figure"), Output("theme-fig", "figure")],
    [Input("bank-drop", "value"), Input("time-window-slider", "value")],
)
def update_overview(bank, months):
    m0, m1 = months
    sel = select(bank, m0, m1)
    period_all = in_range(GLOBAL_DF, m0, m1)
    if sel.empty:
        return dbc.Row([kpi("Complaints", "0")]), empty_fig(), empty_fig()

    shares = sel[THEME_COLS].mean() * 100
    all_shares = period_all[THEME_COLS].mean() * 100
    top_theme = shares.idxmax().replace("theme: ", "")
    kpis = dbc.Row(
        [
            kpi("Complaints", f"{len(sel):,}", f"{100 * len(sel) / len(period_all):.1f}% of all in period"),
            kpi("Median length", f"{int(sel['Words'].median())} words", "per narrative"),
            kpi("Top theme", top_theme, f"{shares.max():.0f}% of complaints"),
            kpi("Credit reporting", f"{shares['theme: Credit reporting']:.0f}%",
                f"all companies: {all_shares['theme: Credit reporting']:.0f}%"),
        ],
        className="g-3",
    )

    # trend: selected company by month (all months), all-company line on a secondary axis
    by_month_bank = GLOBAL_DF[GLOBAL_DF["Company"] == bank].groupby("month_idx").size().reindex(range(N_MONTHS), fill_value=0)
    by_month_all = GLOBAL_DF.groupby("month_idx").size().reindex(range(N_MONTHS), fill_value=0)
    colors = ["#2c82ff" if m0 <= i <= m1 else "#c9d6ea" for i in range(N_MONTHS)]
    trend = go.Figure()
    trend.add_bar(x=MONTHS, y=by_month_bank.values, marker_color=colors, name=bank or "All", hovertemplate="%{x|%b %Y}: %{y}<extra></extra>")
    trend.add_scatter(x=MONTHS, y=by_month_all.values, name="All companies", yaxis="y2", mode="lines", line=dict(color="#f08a4b"))
    trend.update_layout(
        template="plotly_white", margin=dict(t=10, b=30, l=50, r=50), height=340,
        legend=dict(orientation="h", y=1.12), showlegend=True,
        yaxis=dict(title=dict(text="Selected company"), rangemode="tozero"),
        yaxis2=dict(title=dict(text="All companies"), overlaying="y", side="right", showgrid=False, rangemode="tozero"),
    )

    # themes: share of complaints, selected company vs all companies
    order = shares.sort_values().index
    names = [t.replace("theme: ", "") for t in order]
    theme = go.Figure()
    theme.add_bar(
        y=names, x=shares[order].values, orientation="h", name=bank or "All", marker_color="#2c82ff",
        customdata=sel[order].sum().values, hovertemplate="%{y}: %{x:.1f}% (%{customdata} complaints)<extra></extra>",
    )
    theme.add_bar(
        y=names, x=all_shares[order].values, orientation="h", name="All companies", marker_color="#f08a4b",
        hovertemplate="%{y}: %{x:.1f}%<extra>All companies</extra>",
    )
    theme.update_layout(
        template="plotly_white", barmode="group", margin=dict(t=10, b=40, l=130, r=10), height=340,
        xaxis=dict(title=dict(text="% of complaints"), ticksuffix="%"), legend=dict(orientation="h", y=1.12),
    )
    return kpis, trend, theme


@app.callback(
    [
        Output("frequency_figure", "figure"), Output("bank-treemap", "figure"),
        Output("bank-wordcloud", "figure"), Output("no-data-alert", "style"),
    ],
    [Input("bank-drop", "value"), Input("time-window-slider", "value")],
)
def update_word_plots(bank, months):
    figs = word_figures(bank, int(months[0]), int(months[1]))
    if figs is None:
        return empty_fig(), empty_fig(), empty_fig(), {"display": "block"}
    return (*figs, {"display": "none"})


@app.callback(
    [
        Output("lda-table", "data"), Output("lda-table", "columns"),
        Output("tsne-lda", "figure"), Output("no-data-alert-lda", "style"),
    ],
    Input("bank-drop", "value"),
)
def update_lda(bank):
    result = lda_for_bank(bank)
    if result is None:
        return [], [], empty_fig("No precomputed topic model for this company"), {"display": "block"}
    fig, columns, data = result
    return data, columns, fig, {"display": "none"}


@app.callback(
    [Output("lda-table", "filter_query"), Output("lda-table-block", "style")],
    [Input("tsne-lda", "clickData"), Input("bank-drop", "value")],
    State("lda-table", "filter_query"),
)
def filter_table_on_scatter_click(click, _bank, current_filter):
    trigger = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else ""
    if trigger != "tsne-lda" or not click:
        return "", {"display": "none"}
    doc = click["points"][0]["hovertext"]
    query = "{Document_No} eq " + str(doc)
    if current_filter:
        query = "(" + query + ") || (" + current_filter + ")"
    return query, {"display": "block"}


@app.callback(
    Output("bigrams-comps", "figure"),
    [Input("bigrams-comp_1", "value"), Input("bigrams-comp_2", "value")],
)
def comp_bigram_comparisons(first, second):
    d = bigram_df[bigram_df.company.isin([first, second])].copy()
    d["count"] = d["value"]
    if second != first:
        d.loc[d.company == second, "value"] = -d.loc[d.company == second, "value"]
    fig = px.bar(
        d, x="ngram", y="value", color="company", template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold, custom_data=["count"],
        labels={"company": "Company", "ngram": "Bigram"},
    )
    fig.update_traces(hovertemplate="%{x}: %{customdata[0]}<extra>%{fullData.name}</extra>")
    fig.update_layout(legend=dict(orientation="h", x=0.1, y=1.12), height=460, margin=dict(t=30, b=20))
    fig.update_yaxes(title="", showticklabels=False)
    return fig


@app.callback(Output("bigrams-scatter", "figure"), Input("bigrams-perplex-dropdown", "value"))
def populate_bigram_scatter(perplexity):
    d = bigram_embedding(int(perplexity))
    fig = px.scatter(
        d, x="tsne_1", y="tsne_2", hover_name="bigram", text="bigram", size="count", color="words",
        size_max=45, template="plotly_white", labels={"words": "Avg. length<br>(words)"},
        color_continuous_scale=px.colors.sequential.Sunsetdark,
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="Gray")))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=520)
    return fig


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("DASH_DEBUG", "0") == "1",
        port=int(os.environ.get("PORT", 8050)),
    )

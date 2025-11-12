"""
Plot gui for plotting dataset and inspecting it in browswer.
Install bnode with the plotly extension.
"""
"""
BNODE Timeseries Viewer (Plotly/Dash)

Interactive viewer for BNODE HDF5 datasets with time-series (states, outputs, controls)
and optional parameter vectors. This app renders multiple overlaid trajectories per
sample with robust y-scaling and a compact parameters bar chart.

Run
    uv run plot_gui --dataset_path /path/to/dataset.h5 [-n 3] [--no-browser]

Args:
    --dataset_path: Path to the HDF5 dataset. If omitted, you will be prompted.
    -n, --n_trajectory_rows: Number of trajectory rows to display (default: 3).
    --no-browser: Do not auto-open a browser (helpful on remote/headless machines).

Features:
    - White background with black grid lines for clarity.
    - Prev/Next buttons to quickly navigate samples.
    - Robust y-axis scaling using 2.5–97.5 percentiles across all samples, expanded
      to include the selected sample's extrema when they exceed those quantiles.
    - Controls styled in green (solid), truth vs. prediction overlay for states/outputs.
    - Dynamic number of trajectory rows and per-row channel selection.
    - Parameter bar chart (normalized to train-set ranges when available) with values shown.
    - Supports contexts: "train" and/or "common_test".

Expected dataset structure:
    - time: 1D array of length T (seconds; auto-shown in hours if very large magnitudes).
    - contexts: "train" and/or "common_test"
        - states, outputs, controls: float arrays shaped (N, C, T)
        - states_hat, outputs_hat: optional predictions shaped (N, C, T)
        - parameters: optional (N, P)
    - *_names: optional per-channel names datasets (e.g., states_names, outputs_names, controls_names)
    - parameters_names: optional per-parameter names

Notes:
    - Requires Dash; install bnode-core using the bnode-core[plotly] option.
    - Provided as a console script entry point named `plot_gui` so it can be invoked with `uv run plot_gui`.
    - For remote sessions, run with `--no-browser` and open the URL printed in the console.

Example:
    uv run plot_gui --dataset_path ./data/my_dataset.hdf5 -n 4
"""

from pathlib import Path
import h5py
import numpy as np
import argparse
from typing import List, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from dash import Dash, dcc, html, Input, Output, State
    import dash
    _HAS_DASH = True
except Exception:
    _HAS_DASH = False


def _detect_time_axis(f: h5py.File) -> Tuple[np.ndarray, str]:
    time = f['time'][:]
    if np.max(time) > 1.0e4:
        return time / 3600.0, 'time [h]'
    return time, 'time [s]'


def _available_contexts(f: h5py.File) -> List[str]:
    return [c for c in ['train', 'common_test'] if c in f]


def _build_channel_options(f: h5py.File) -> List[Tuple[str, Tuple[str, int]]]:
    options = []
    for dtype in ['states', 'outputs', 'controls']:
        if 'train' in f and dtype in f['train']:
            names_key = f"{dtype}_names"
            names = np.array(f[names_key][:], dtype='str') if names_key in f else np.array([f"{dtype}_{i}" for i in range(f['train'][dtype].shape[1])])
            for ch in range(len(names)):
                label = f"{dtype}:{names[ch]}"
                options.append((label, (dtype, ch)))
    return options


def _has_hat(f: h5py.File, context: str, dtype: str) -> bool:
    key = f"{dtype}_hat"
    return key in f[context]


def _get_sample_count(f: h5py.File, context: str) -> int:
    # prefer states if available; otherwise outputs; otherwise controls
    for dtype in ['states', 'outputs', 'controls']:
        if dtype in f[context]:
            return f[context][dtype].shape[0]
    return 0


def _get_series(f: h5py.File, context: str, dtype: str, ch: int, sidx: int) -> Tuple[np.ndarray, np.ndarray]:
    y = f[context][dtype][sidx, ch, :]
    y_hat = None
    if dtype != 'controls' and _has_hat(f, context, dtype):
        y_hat = f[context][f"{dtype}_hat"][sidx, ch, :]
    return y, y_hat


def _quantile_bounds(f: h5py.File, context: str, dtype: str, ch: int, lower_q: float = 2.5, upper_q: float = 97.5) -> Tuple[float, float]:
    """Compute lower/upper quantile across all samples and timesteps for one channel.

    We flatten (samples, time) for the given context/dtype/channel to get robust bounds.
    If dtype not present, return (None, None).
    """
    if dtype not in f[context]:
        return None, None
    # shape: (samples, channels, time) -> select channel then flatten samples & time
    arr = f[context][dtype][:, ch, :].reshape(-1)
    low = float(np.percentile(arr, lower_q))
    high = float(np.percentile(arr, upper_q))
    if not np.isfinite(low) or not np.isfinite(high):
        return None, None
    if low == high:
        # expand a little to make visible range
        low -= 1e-6
        high += 1e-6
    return low, high


def _get_label(f: h5py.File, dtype: str, ch: int) -> str:
    names_key = f"{dtype}_names"
    if names_key in f:
        names = np.array(f[names_key][:], dtype='str')
        if ch < len(names):
            return names[ch]
    return f"{dtype}[{ch}]"


def _get_parameters(f: h5py.File, context: str, sidx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if 'parameters' not in f[context]:
        return None, None, None
    names = np.array(f['parameters_names'][:], dtype='str') if 'parameters_names' in f else np.array([f"p{i}" for i in range(f[context]['parameters'].shape[1])])
    params = f[context]['parameters'][sidx, :]
    # normalization based on train set if available
    if 'train' in f and 'parameters' in f['train']:
        mins = f['train']['parameters'][:].min(axis=0)
        maxs = f['train']['parameters'][:].max(axis=0)
        denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
        params_norm = (params - mins) / denom
    else:
        params_norm = params
    return names, params, params_norm


def plot_gui(dataset_path: Path, n_trajectory_rows: int = 3):
    if not _HAS_DASH:
        raise RuntimeError("Dash is required for the Plotly GUI. Please install with `pip install dash`. ")

    f = h5py.File(dataset_path, 'r')
    time, time_label = _detect_time_axis(f)
    contexts = _available_contexts(f)
    if not contexts:
        raise ValueError("No supported contexts found in dataset (expected 'train' and/or 'common_test').")
    channel_options = _build_channel_options(f)

    # Pre-create dropdowns up to a reasonable maximum; user can choose how many plots to display
    MAX_ROWS = min(8, max(1, len(channel_options)))
    # defaults for each row: first channels
    default_rows = [channel_options[min(i, len(channel_options)-1)][1] for i in range(MAX_ROWS)]

    app = Dash(__name__)

    def dropdown_options(opts):
        return [{'label': lbl, 'value': f"{dt}|{ch}"} for (lbl, (dt, ch)) in opts]

    def parse_value(val: str) -> Tuple[str, int]:
        dt, ch = val.split('|')
        return dt, int(ch)

    context_default = contexts[0]
    sample_default_count = _get_sample_count(f, context_default)
    sample_default = 0 if sample_default_count > 0 else None

    controls = [
        html.Div([
            html.Label('Context'),
            dcc.Dropdown(id='context-dd', options=[{'label': c, 'value': c} for c in contexts], value=context_default, clearable=False)
        ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.Label('Sample index'),
            html.Div([
                html.Button('Prev', id='prev-btn', n_clicks=0, style={'marginRight': '0.5rem'}),
                dcc.Dropdown(id='sample-dd', options=[{'label': f"{context_default} #{i}", 'value': i} for i in range(sample_default_count)], value=sample_default, clearable=False, style={'width': '8rem', 'display': 'inline-block', 'verticalAlign': 'middle'}),
                html.Button('Next', id='next-btn', n_clicks=0, style={'marginLeft': '0.5rem'})
            ])
        ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1rem'}),
        html.Div([
            html.Label('Number of plots'),
            dcc.Dropdown(id='nplots-dd', options=[{'label': str(i), 'value': i} for i in range(1, MAX_ROWS+1)], value=min(n_trajectory_rows, MAX_ROWS), clearable=False)
        ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1rem'}),
    ]

    # One trajectory dropdown per row
    for i in range(MAX_ROWS):
        controls.append(
            html.Div([
                html.Label(f'Trajectory row {i+1}'),
                dcc.Dropdown(id=f'traj-dd-{i}', options=dropdown_options(channel_options),
                             value=f"{default_rows[i][0]}|{default_rows[i][1]}", clearable=False)
            ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1rem', 'marginBottom': '0.75rem'})
        )

    app.layout = html.Div([
        html.H3('BNODE Timeseries Viewer (Plotly)'),
        dcc.Store(id='sample-index', data=sample_default or 0),
        html.Div(controls),
        dcc.Graph(id='timeseries-graph'),
        dcc.Graph(id='parameters-graph')
    ])

    # Update sample dropdown options when context changes
    @app.callback(Output('sample-dd', 'options'), Input('context-dd', 'value'))
    def _update_sample_dd_options(context):
        count = _get_sample_count(f, context)
        opts = [{'label': f"{context} #{i}", 'value': i} for i in range(count)]
        return opts

    # Maintain current sample index in a Store and update via Prev/Next buttons or context resets
    @app.callback(Output('sample-index', 'data'),
                  Input('prev-btn', 'n_clicks'),
                  Input('next-btn', 'n_clicks'),
                  Input('context-dd', 'value'),
                  State('sample-index', 'data'),
                  State('sample-dd', 'options'))
    def _navigate_samples(prev_clicks, next_clicks, context, current_idx, options):
        # Determine how many samples available under current context
        count = len(options) if options is not None else _get_sample_count(f, context)
        if count is None:
            count = 0
        if current_idx is None:
            current_idx = 0
        # Which input triggered
        trig = dash.callback_context.triggered[0]['prop_id'].split('.')[0] if dash.callback_context.triggered else None
        if trig == 'context-dd':
            return 0 if count > 0 else None
        if trig == 'prev-btn':
            return max(0, int(current_idx) - 1) if count > 0 else None
        if trig == 'next-btn':
            return min(max(0, count - 1), int(current_idx) + 1) if count > 0 else None
        # Initial load
        return 0 if count > 0 else None

    # Sync the dropdown value from the store
    @app.callback(Output('sample-dd', 'value'), Input('sample-index', 'data'), State('sample-dd', 'options'))
    def _sync_sample_value(idx, options):
        if idx is None:
            return None
        count = len(options) if options is not None else 0
        if count == 0:
            return None
        # Clamp
        idx = max(0, min(int(idx), count - 1))
        return idx

    # Update figures when any selection changes
    inputs = [Input('context-dd', 'value'), Input('sample-dd', 'value'), Input('nplots-dd', 'value')] + [Input(f'traj-dd-{i}', 'value') for i in range(MAX_ROWS)]

    @app.callback(Output('timeseries-graph', 'figure'), Output('parameters-graph', 'figure'), inputs)
    def _update_fig(context, sidx, nplots, *traj_vals):
        if sidx is None:
            return go.Figure(), go.Figure()
        # Build timeseries subplot figure
        rows = int(max(1, nplots))
        # Prepare subplot titles with variable names
        titles = []
        parsed = []
        for i in range(rows):
            dtype, ch = parse_value(traj_vals[i])
            parsed.append((dtype, ch))
            titles.append(_get_label(f, dtype, ch))
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=titles)
        for i in range(rows):
            dtype, ch = parsed[i]
            y, y_hat = _get_series(f, context, dtype, ch, sidx)
            # Truth styling: controls -> green solid, others -> red dashed
            if dtype == 'controls':
                truth_line_style = dict(color='green')
                truth_line_mode = 'lines'
            else:
                truth_line_style = dict(color='red', dash='dash')
                truth_line_mode = 'lines'
            fig.add_trace(go.Scatter(x=time, y=y, name=f"{dtype}[{ch}]", mode=truth_line_mode, line=truth_line_style),
                          row=i+1, col=1)
            # Prediction (if available and not controls)
            if y_hat is not None:
                fig.add_trace(go.Scatter(x=time, y=y_hat, name=f"{dtype}_hat[{ch}]", mode='lines', line=dict(color='blue')),
                              row=i+1, col=1)
            fig.update_yaxes(title_text=f"{dtype}[{ch}]", row=i+1, col=1)
            # Quantile-based y-axis limits per subplot
            q_low, q_high = _quantile_bounds(f, context, dtype, ch)
            # incorporate selected-sample extrema if outside quantile bounds
            sample_min = float(np.nanmin(y)) if y is not None else None
            sample_max = float(np.nanmax(y)) if y is not None else None
            if y_hat is not None:
                sample_min = float(np.nanmin([sample_min, np.nanmin(y_hat)])) if sample_min is not None else float(np.nanmin(y_hat))
                sample_max = float(np.nanmax([sample_max, np.nanmax(y_hat)])) if sample_max is not None else float(np.nanmax(y_hat))
            # Determine final bounds
            if q_low is None or q_high is None:
                low, high = sample_min, sample_max
            else:
                low = min(q_low, sample_min) if sample_min is not None else q_low
                high = max(q_high, sample_max) if sample_max is not None else q_high
            if low is not None and high is not None and np.isfinite(low) and np.isfinite(high):
                if low == high:
                    low -= 1e-6
                    high += 1e-6
                pad = 0.05 * (high - low)
                fig.update_yaxes(row=i+1, col=1, range=[low - pad, high + pad])
        fig.update_xaxes(title_text=time_label, row=rows, col=1)
        fig.update_layout(
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0),
            margin=dict(l=40, r=20, t=40, b=40),
            height=260*rows,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, 'Open Sans', Roboto, sans-serif", size=13)
        )
        # Black grid lines on white background
        fig.update_xaxes(showgrid=True, gridcolor='black')
        fig.update_yaxes(showgrid=True, gridcolor='black')

        # Parameters bar chart (if available)
        pfig = go.Figure()
        names, params, params_norm = _get_parameters(f, context, sidx)
        if names is not None:
            pfig.add_bar(x=names, y=params_norm, name='normalized')
            # show true values as text to avoid second axis for speed
            pfig.update_traces(text=[f"{v:.2f}" for v in params], textposition='outside')
            pfig.update_yaxes(range=[0, 1.1], title_text='parameters (normalized)')
            pfig.update_layout(margin=dict(l=40, r=20, t=40, b=80), height=300, template='plotly_white',
                               plot_bgcolor='white', paper_bgcolor='white',
                               font=dict(family="Inter, 'Open Sans', Roboto, sans-serif", size=13))
            # Black grid lines for parameters chart
            pfig.update_xaxes(showgrid=True, gridcolor='black')
            pfig.update_yaxes(showgrid=True, gridcolor='black')
        return fig, pfig

    # Run app
    app.run(debug=False)


# add parser for file path
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, help='path to dataset', default=None)
parser.add_argument('-n', '--n_trajectory_rows', type=int, help='number of trajectory rows to plot', default=3)
parser.add_argument('--no-browser', action='store_true', help='Do not auto-open a browser (headless server).')


def main():
    args = parser.parse_args()
    if args.dataset_path is None:
        args.dataset_path = input('Enter path to dataset: ')
    if args.dataset_path.startswith('file://'):
        _path = Path(args.dataset_path.replace('file://',''))
    else:
        _path = Path(args.dataset_path)
        if not _path.exists():
            raise FileNotFoundError(f"The dataset path {args.dataset_path} can not be opened.")
    print(f'Opening dataset at {_path}')
    plot_gui(_path, n_trajectory_rows=args.n_trajectory_rows)


if __name__ == '__main__':
    main()
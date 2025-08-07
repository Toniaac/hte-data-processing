import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from sklearn.linear_model import LinearRegression
import pandas as pd


data_grouped = pd.read_csv('all_data_grouped_for_analysis.csv')

# --- Color-blind-friendly palette (Okabe–Ito) ---
palette = {
    "high_RH": "#0072B2",     # blue
    "low_RH":  "#D55E00",     # vermilion
    "dry_N2":  "#009E73",     # bluish green
}

# Define humidity cutoff and categorize data
RH_cutoff = 49
high_RH = data_grouped[data_grouped['Relative Humidity'] >= RH_cutoff]
low_RH = data_grouped[data_grouped['Relative Humidity'] < RH_cutoff]
dry_nitrogen = data_grouped[data_grouped['Nitrogen (Side from drop)'] == 1]

plt.figure(figsize=(8, 6))

# Styling
marker_size = 15
cap_size = 3
error_line_width = 1.5
fit_line_alpha = 0.7
fit_line_style = '--'
marker_face_alpha = 0.55       # << marker fill transparency
marker_edge_alpha = 0.9        # << marker edge transparency
marker_edge_width = 1.8

# High RH
plt.errorbar(
    high_RH['Concentration'], high_RH['Elastic Modulus_mean'],
    yerr=high_RH['Elastic Modulus_std'], fmt='o',
    color=palette["high_RH"],                 # line color
    ecolor=palette["high_RH"],                # errorbar color
    markerfacecolor=to_rgba(palette["high_RH"], marker_face_alpha),
    markeredgecolor=to_rgba(palette["high_RH"], marker_edge_alpha),
    markeredgewidth=marker_edge_width, markersize=marker_size,
    capsize=cap_size, elinewidth=error_line_width,
    label=f'RH ≥ {RH_cutoff}%'
)

# Low RH
plt.errorbar(
    low_RH['Concentration'], low_RH['Elastic Modulus_mean'],
    yerr=low_RH['Elastic Modulus_std'], fmt='s',
    color=palette["low_RH"], ecolor=palette["low_RH"],
    markerfacecolor=to_rgba(palette["low_RH"], marker_face_alpha),
    markeredgecolor=to_rgba(palette["low_RH"], marker_edge_alpha),
    markeredgewidth=marker_edge_width, markersize=marker_size,
    capsize=cap_size, elinewidth=error_line_width,
    label=f'RH < {RH_cutoff}%'
)

# Dry N2
plt.errorbar(
    dry_nitrogen['Concentration'], dry_nitrogen['Elastic Modulus_mean'],
    yerr=dry_nitrogen['Elastic Modulus_std'], fmt='^',
    color=palette["dry_N2"], ecolor=palette["dry_N2"],
    markerfacecolor=to_rgba(palette["dry_N2"], marker_face_alpha),
    markeredgecolor=to_rgba(palette["dry_N2"], marker_edge_alpha),
    markeredgewidth=marker_edge_width, markersize=marker_size,
    capsize=cap_size, elinewidth=error_line_width,
    label='Dry Nitrogen'
)

# Fit lines (keep separate alpha control)
def fit_and_extrapolate(data, color, x_min, x_max):
    if len(data) > 1:
        model = LinearRegression()
        X = data[['Concentration']].values
        y = data['Elastic Modulus_mean'].values
        model.fit(X, y)
        x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        plt.plot(x_range, y_pred, linestyle=fit_line_style, color=color, alpha=fit_line_alpha)

x_min, x_max = data_grouped['Concentration'].min(), data_grouped['Concentration'].max()
fit_and_extrapolate(high_RH, palette["high_RH"], x_min, x_max)
fit_and_extrapolate(low_RH, palette["low_RH"], x_min, x_max)
fit_and_extrapolate(dry_nitrogen, palette["dry_N2"], x_min, x_max)

plt.xlabel('Polymer Concentration (wt%)', fontsize=18)
plt.ylabel('Elastic Modulus (bar)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16, title='Humidity Condition', title_fontsize=16)
plt.tight_layout()
plt.show()

# --- Color-blind-friendly palette (Okabe–Ito) ---
palette = {
    "high_RH": "#0072B2",   # blue
    "low_RH":  "#D55E00",   # vermilion
    "dry_N2":  "#009E73",   # bluish green
}

plt.figure(figsize=(8, 6))

# Styling parameters
marker_size = 15
cap_size = 3
error_line_width = 1.5
fit_line_alpha = 0.7
fit_line_style = '--'
marker_edge_width = 1.8
marker_face_alpha = 0.55
marker_edge_alpha = 0.9

# High RH
plt.errorbar(
    high_RH['Concentration'], high_RH['Changepoint_mean'],
    yerr=high_RH['Changepoint_std'], fmt='o',
    color=palette["high_RH"], ecolor=palette["high_RH"],
    markerfacecolor=to_rgba(palette["high_RH"], marker_face_alpha),
    markeredgecolor=to_rgba(palette["high_RH"], marker_edge_alpha),
    markeredgewidth=marker_edge_width, markersize=marker_size,
    capsize=cap_size, elinewidth=error_line_width,
    label=f'RH ≥ {RH_cutoff}%'
)

# Low RH
plt.errorbar(
    low_RH['Concentration'], low_RH['Changepoint_mean'],
    yerr=low_RH['Changepoint_std'], fmt='s',
    color=palette["low_RH"], ecolor=palette["low_RH"],
    markerfacecolor=to_rgba(palette["low_RH"], marker_face_alpha),
    markeredgecolor=to_rgba(palette["low_RH"], marker_edge_alpha),
    markeredgewidth=marker_edge_width, markersize=marker_size,
    capsize=cap_size, elinewidth=error_line_width,
    label=f'RH < {RH_cutoff}%'
)

# Dry N2
plt.errorbar(
    dry_nitrogen['Concentration'], dry_nitrogen['Changepoint_mean'],
    yerr=dry_nitrogen['Changepoint_std'], fmt='^',
    color=palette["dry_N2"], ecolor=palette["dry_N2"],
    markerfacecolor=to_rgba(palette["dry_N2"], marker_face_alpha),
    markeredgecolor=to_rgba(palette["dry_N2"], marker_edge_alpha),
    markeredgewidth=marker_edge_width, markersize=marker_size,
    capsize=cap_size, elinewidth=error_line_width,
    label='Dry Nitrogen'
)

# Fit + extrapolate
def fit_and_extrapolate_changepoint(data, color, x_min, x_max):
    if len(data) > 1:
        model = LinearRegression()
        X = data[['Concentration']].values
        y = data['Changepoint_mean'].values
        model.fit(X, y)
        x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        plt.plot(x_range, y_pred, color=color, linestyle=fit_line_style, alpha=fit_line_alpha)

x_min = data_grouped['Concentration'].min()
x_max = data_grouped['Concentration'].max()
fit_and_extrapolate_changepoint(high_RH, palette["high_RH"], x_min, x_max)
fit_and_extrapolate_changepoint(low_RH,  palette["low_RH"],  x_min, x_max)
fit_and_extrapolate_changepoint(dry_nitrogen, palette["dry_N2"], x_min, x_max)

plt.xlabel('Polymer Concentration (wt%)', fontsize=18)
plt.ylabel('Pore Fraction', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16, title='Humidity Condition', title_fontsize=16)
plt.tight_layout()
plt.show()

#!/usr/bin/env python3
"""
Ring Magnet Force & Field Visualizer - Streamlit Web App
Uses pre-computed FEMM data and magpylib for field visualization.
"""

import streamlit as st
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import magpylib as magpy

# Page config
st.set_page_config(
    page_title="Ring Magnet Force Calculator",
    page_icon="🧲",
    layout="wide"
)

# N42 NdFeB properties
BR_N42 = 1.30  # Tesla (remanence)

# Path to pre-computed FEMM sweep data
SWEEP_FILE = 'sweep_results.json'


@st.cache_data
def load_sweep_data():
    """Load pre-computed FEMM sweep results."""
    sweep_data = {}
    force_lookup = {}

    if os.path.exists(SWEEP_FILE):
        with open(SWEEP_FILE, 'r') as f:
            sweep_data = json.load(f)

        # Build lookup dictionary
        for key, data in sweep_data.items():
            lookup_key = (
                data['inner_diameter'],
                data['outer_diameter'],
                data['thickness'],
                data['gap']
            )
            force_lookup[lookup_key] = data['force_N']

        # Extract unique parameter values
        id_values = sorted(set(v['inner_diameter'] for v in sweep_data.values()))
        thickness_values = sorted(set(v['thickness'] for v in sweep_data.values()))
        gap_values = sorted(set(v['gap'] for v in sweep_data.values()))

        # Build ID -> valid ODs mapping
        id_to_ods = {}
        for v in sweep_data.values():
            id_val = v['inner_diameter']
            od_val = v['outer_diameter']
            if id_val not in id_to_ods:
                id_to_ods[id_val] = set()
            id_to_ods[id_val].add(od_val)
        for id_val in id_to_ods:
            id_to_ods[id_val] = sorted(id_to_ods[id_val])

        return sweep_data, force_lookup, id_values, thickness_values, gap_values, id_to_ods
    else:
        st.error(f"Data file not found: {SWEEP_FILE}")
        return {}, {}, [], [], [], {}


def get_force(force_lookup, id_val, od_val, thickness, gap):
    """Look up force from pre-computed data."""
    key = (id_val, od_val, thickness, gap)
    return force_lookup.get(key, None)


def get_force_curve(force_lookup, gap_values, id_val, od_val, thickness):
    """Get force vs gap curve from actual FEMM data points."""
    gaps = []
    forces = []
    for gap in gap_values:
        f = get_force(force_lookup, id_val, od_val, thickness, gap)
        if f is not None:
            gaps.append(gap)
            forces.append(f)
    return gaps, forces


def create_ring_magnet(inner_d_mm, outer_d_mm, thickness_mm, z_position_mm=0):
    """Create a ring magnet using magpylib."""
    inner_r = inner_d_mm / 2 / 1000
    outer_r = outer_d_mm / 2 / 1000
    h = thickness_mm / 1000
    z_pos = (z_position_mm + thickness_mm / 2) / 1000

    MU_0 = 4 * np.pi * 1e-7
    M_Am = BR_N42 / MU_0

    magnet = magpy.magnet.CylinderSegment(
        magnetization=(0, 0, M_Am),
        dimension=(inner_r, outer_r, h, 0, 360),
        position=(0, 0, z_pos)
    )
    return magnet


@st.cache_data
def calculate_field(id_val, od_val, thickness, gap):
    """Calculate magnetic field on a 2D grid."""
    inner_r = id_val / 2
    outer_r = od_val / 2

    mag1 = create_ring_magnet(id_val, od_val, thickness, 0)
    mag2 = create_ring_magnet(id_val, od_val, thickness, thickness + gap)
    collection = magpy.Collection(mag1, mag2)

    r_max = outer_r * 1.5
    z_max = 2 * thickness + gap + thickness * 0.3
    z_min = -thickness * 0.2

    n_r, n_z = 60, 80
    r_vals_mm = np.linspace(0.5, r_max, n_r)
    z_vals_mm = np.linspace(z_min, z_max, n_z)

    R_mm, Z_mm = np.meshgrid(r_vals_mm, z_vals_mm)
    R_m = R_mm / 1000
    Z_m = Z_mm / 1000

    points_m = np.column_stack([R_m.ravel(), np.zeros(R_m.size), Z_m.ravel()])
    B_vectors = collection.getB(points_m)
    B_mag = np.linalg.norm(B_vectors, axis=1).reshape(R_mm.shape)

    return R_mm, Z_mm, B_mag, inner_r, outer_r


def plot_geometry(id_val, od_val, thickness, gap):
    """Create geometry cross-section plot."""
    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax.set_title("Cross-Section", fontsize=9)
    ax.set_xlabel("r (mm)", fontsize=8)
    ax.set_ylabel("z (mm)", fontsize=8)
    ax.tick_params(labelsize=7)

    inner_r = id_val / 2
    outer_r = od_val / 2

    # Magnet 1 (bottom)
    mag1 = patches.Rectangle((inner_r, 0), outer_r - inner_r, thickness,
                               linewidth=1.5, edgecolor='darkblue', facecolor='royalblue', alpha=0.8)
    ax.add_patch(mag1)

    # Magnet 2 (top)
    mag2 = patches.Rectangle((inner_r, thickness + gap), outer_r - inner_r, thickness,
                               linewidth=1.5, edgecolor='darkred', facecolor='indianred', alpha=0.8)
    ax.add_patch(mag2)

    # N/S labels
    mid_r = (inner_r + outer_r) / 2
    if thickness > 8:
        ax.annotate('N', (mid_r, thickness - 2), ha='center', fontsize=7, fontweight='bold', color='white')
        ax.annotate('S', (mid_r, 2), ha='center', fontsize=7, fontweight='bold', color='white')
        ax.annotate('S', (mid_r, thickness + gap + 2), ha='center', fontsize=7, fontweight='bold', color='white')
        ax.annotate('N', (mid_r, 2 * thickness + gap - 2), ha='center', fontsize=7, fontweight='bold', color='white')

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    max_z = 2 * thickness + gap + 5
    ax.set_xlim(-2, outer_r + 10)
    ax.set_ylim(-2, max_z)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()

    return fig


def plot_force_curve(gaps, forces, current_gap, current_force):
    """Create force vs gap plot."""
    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax.set_title("Force vs Gap", fontsize=9)
    ax.set_xlabel("Gap (mm)", fontsize=8)
    ax.set_ylabel("Force (N)", fontsize=8)
    ax.tick_params(labelsize=7)

    if gaps and forces:
        ax.plot(gaps, forces, 'b-', linewidth=1.5)
        ax.plot(gaps, forces, 'bo', markersize=5, label='FEMM data')

        if current_force and current_force > 0:
            ax.plot(current_gap, current_force, 'ro', markersize=8,
                    label=f'{current_force:.0f} N', zorder=5)

        ax.legend(fontsize=6)
        ax.set_ylim(0, max(forces) * 1.1)

    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim(0, 27)
    fig.tight_layout()

    return fig


def plot_field(R_mm, Z_mm, B_mag, inner_r, outer_r, thickness, gap):
    """Create magnetic field contour plot."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.set_title("Magnetic Field |B|", fontsize=9)
    ax.set_xlabel("r (mm)", fontsize=8)
    ax.set_ylabel("z (mm)", fontsize=8)
    ax.tick_params(labelsize=7)

    B_plot = np.clip(B_mag, 0, 2.0)
    im = ax.pcolormesh(R_mm, Z_mm, B_plot, cmap='hot', shading='auto',
                       vmin=0, vmax=min(1.5, np.max(B_plot) * 1.1))

    cbar = plt.colorbar(im, ax=ax, label='T')
    cbar.ax.tick_params(labelsize=6)

    # Magnet outlines
    ax.plot([inner_r, outer_r, outer_r, inner_r, inner_r],
            [0, 0, thickness, thickness, 0], 'c-', linewidth=1.5)
    ax.plot([inner_r, outer_r, outer_r, inner_r, inner_r],
            [thickness + gap, thickness + gap, 2 * thickness + gap, 2 * thickness + gap, thickness + gap],
            'c-', linewidth=1.5)

    ax.set_xlim(0, R_mm.max())
    ax.set_ylim(Z_mm.min(), Z_mm.max())
    fig.tight_layout()

    return fig


# Main app
def main():
    st.title("🧲 Ring Magnet Force Calculator")

    # Load data
    sweep_data, force_lookup, id_values, thickness_values, gap_values, id_to_ods = load_sweep_data()

    if not sweep_data:
        st.stop()

    # Sidebar controls
    st.sidebar.header("Magnet Parameters")

    id_val = st.sidebar.selectbox("Inner Diameter (mm)", id_values, index=1)

    valid_ods = id_to_ods.get(id_val, [])
    od_val = st.sidebar.selectbox("Outer Diameter (mm)", valid_ods, index=0 if valid_ods else 0)

    thickness = st.sidebar.selectbox("Thickness (mm)", thickness_values, index=0)
    gap = st.sidebar.selectbox("Gap (mm)", gap_values, index=1)

    # Calculate force
    force = get_force(force_lookup, id_val, od_val, thickness, gap)

    # Display force prominently
    st.sidebar.markdown("---")
    st.sidebar.header("Attraction Force")
    if force:
        force_lbs = force * 0.224809
        force_kg = force / 9.80665
        st.sidebar.metric("Force", f"{force:.1f} N")
        st.sidebar.write(f"**{force_lbs:.2f} lbs** / **{force_kg:.2f} kg**")
    else:
        st.sidebar.warning("No data for this combination")

    # Material info
    st.sidebar.markdown("---")
    st.sidebar.caption(f"**Material:** N42 NdFeB (Br = {BR_N42} T)")
    st.sidebar.caption(f"**Data points:** {len(sweep_data)}")

    # Main content - all 3 plots in a row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.pyplot(plot_geometry(id_val, od_val, thickness, gap))

    with col2:
        gaps, forces = get_force_curve(force_lookup, gap_values, id_val, od_val, thickness)
        st.pyplot(plot_force_curve(gaps, forces, gap, force))

    with col3:
        with st.spinner("Calculating..."):
            R_mm, Z_mm, B_mag, inner_r, outer_r = calculate_field(id_val, od_val, thickness, gap)
            st.pyplot(plot_field(R_mm, Z_mm, B_mag, inner_r, outer_r, thickness, gap))

    # Configuration summary (compact)
    wall = (od_val - id_val) / 2
    volume = np.pi * ((od_val / 2) ** 2 - (id_val / 2) ** 2) * thickness / 1000
    st.caption(f"Wall: {wall:.0f}mm | Volume: {volume:.1f}cm³ | Gap: {gap}mm")


if __name__ == '__main__':
    main()

"""
Interactive plotting functions for spectral analysis.

Provides modern, interactive visualizations using Plotly and ipywidgets.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import ipywidgets as widgets
    from IPython.display import display
    from ipywidgets import Button, FloatSlider, FloatText, HBox, Output, VBox
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False

# Import create_lorentzian_spectrum function
try:
    from .spectral import create_lorentzian_spectrum
except ImportError:
    # Fallback if not available
    def create_lorentzian_spectrum(lines, energy_grid):
        """Fallback Lorentzian spectrum calculation"""
        spectrum = np.zeros_like(energy_grid)
        for _, line in lines.iterrows():
            # Simple Lorentzian: amplitude / ((x - center)^2 + (width/2)^2)
            width = getattr(line, 'line_width', 1.0)  # Default width if not available
            lorentzian = line.intensity_final / ((energy_grid - line.energy)**2 + (width/2)**2)
            spectrum += lorentzian
        return spectrum


def create_interactive_energy_shifter(
    diagram_lines: pd.DataFrame,
    satellite_lines: Optional[pd.DataFrame] = None,
    lorentzian_spectrum: Optional[np.ndarray] = None,
    energy_grid: Optional[np.ndarray] = None,
    shell: str = "1s",
    energy_min: float = 20500,
    energy_max: float = 21800
) -> VBox:
    """
    Create an interactive Plotly plot for energy shifting of spectral lines.
    
    Allows real-time adjustment of diagram and satellite line energies with sliders,
    plus adjustable x-axis range. Includes text inputs linked to sliders for precise values.
    
    Parameters
    ----------
    diagram_lines : pd.DataFrame
        Diagram lines with 'energy', 'intensity_final' columns
    satellite_lines : pd.DataFrame, optional
        Satellite lines with 'energy', 'intensity_final' columns
    lorentzian_spectrum : np.ndarray, optional
        Pre-computed Lorentzian spectrum
    energy_grid : np.ndarray, optional
        Energy grid for Lorentzian spectrum
    shell : str
        Shell name for plot titles (default: "1s")
    energy_min : float
        Initial minimum energy for x-axis (default: 20500)
    energy_max : float
        Initial maximum energy for x-axis (default: 21800)
        
    Returns
    -------
    ipywidgets.VBox
        Interactive widget containing plot and controls
        
    Notes
    -----
    Requires ipywidgets and plotly to be installed.
    If lorentzian_spectrum and energy_grid are provided, shows broadened spectrum.
    """
    if not IPYWIDGETS_AVAILABLE:
        raise ImportError("ipywidgets is required for interactive plotting")
    
    # Create widgets with better layout
    diagram_slider = FloatSlider(min=-50, max=50, step=0.1, value=0, 
                                description='Diagram Shift (eV):',
                                layout=widgets.Layout(width='300px'))
    satellite_slider = FloatSlider(min=-50, max=50, step=0.1, value=0, 
                                  description='Satellite Shift (eV):',
                                  layout=widgets.Layout(width='300px'))
    x_min_slider = FloatSlider(min=19000, max=22000, step=10, value=energy_min, 
                              description='X Min (eV):',
                              layout=widgets.Layout(width='300px'))
    x_max_slider = FloatSlider(min=19000, max=22000, step=10, value=energy_max, 
                              description='X Max (eV):',
                              layout=widgets.Layout(width='300px'))
    
    diagram_text = FloatText(value=0, description='Diagram (eV):', 
                           layout=widgets.Layout(width='150px'))
    satellite_text = FloatText(value=0, description='Satellite (eV):', 
                             layout=widgets.Layout(width='150px'))
    
    # Link sliders and text inputs
    widgets.jslink((diagram_slider, 'value'), (diagram_text, 'value'))
    widgets.jslink((satellite_slider, 'value'), (satellite_text, 'value'))
    
    # Reset button
    reset_button = Button(description='Reset to 0', button_style='warning')
    
    def reset_shifts(b):
        diagram_slider.value = 0
        satellite_slider.value = 0
    
    reset_button.on_click(reset_shifts)
    
    # Output widget for plot with better sizing
    output = Output(layout=widgets.Layout(height='700px', width='100%'))
    
    # Text output for displaying current values
    value_output = Output(layout=widgets.Layout(height='100px', width='100%'))
    
    def update_plot(diagram_shift=0.0, satellite_shift=0.0, x_min=energy_min, x_max=energy_max):
        """Update the plot by recreating it with new values"""
        with output:
            output.clear_output(wait=True)
            
            # Shift dataframes
            diagram_shifted = diagram_lines.copy()
            diagram_shifted['energy'] += diagram_shift
            
            satellite_shifted = satellite_lines.copy() if satellite_lines is not None else None
            if satellite_shifted is not None:
                satellite_shifted['energy'] += satellite_shift
            
            # Create energy grid for current range
            energy_grid_local = np.linspace(x_min, x_max, 2000)
            
            # Create subplots with increased spacing
            fig = make_subplots(rows=2, cols=1, 
                               subplot_titles=(f'{shell}-shell Spectral Lines (Delta Functions)', 
                                               f'{shell}-shell Lorentzian-Broadened Spectrum (Natural Widths)'),
                               vertical_spacing=0.25)  # Increased spacing
            
            # Plot diagram lines as stems
            if len(diagram_shifted) > 0:
                x_stem = []
                y_stem = []
                for _, row in diagram_shifted.iterrows():
                    x_stem.extend([row['energy'], row['energy'], None])
                    y_stem.extend([0, row['intensity_final'], None])
                fig.add_trace(go.Scatter(x=x_stem, y=y_stem, mode='lines', 
                                         line=dict(color='coral', width=1), 
                                         name=f'Diagram Lines (shift: {diagram_shift:.1f} eV)'),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=diagram_shifted['energy'], 
                                         y=diagram_shifted['intensity_final'],
                                         mode='markers',
                                         marker=dict(color='coral', size=6, symbol='circle'),
                                         showlegend=False),
                              row=1, col=1)
            
            # Plot satellite lines as stems
            if satellite_shifted is not None and len(satellite_shifted) > 0:
                x_stem = []
                y_stem = []
                for _, row in satellite_shifted.iterrows():
                    x_stem.extend([row['energy'], row['energy'], None])
                    y_stem.extend([0, row['intensity_final'], None])
                fig.add_trace(go.Scatter(x=x_stem, y=y_stem, mode='lines', 
                                         line=dict(color='skyblue', width=1), 
                                         name=f'Satellite Lines (shift: {satellite_shift:.1f} eV)'),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=satellite_shifted['energy'], 
                                         y=satellite_shifted['intensity_final'],
                                         mode='markers',
                                         marker=dict(color='skyblue', size=4, symbol='diamond'),
                                         showlegend=False),
                              row=1, col=1)
            
            # Plot Lorentzian spectrum if provided
            if lorentzian_spectrum is not None and len(lorentzian_spectrum) > 0 and energy_grid is not None:
                # Recalculate Lorentzian spectrum from shifted lines
                shifted_lorentzian = np.zeros_like(energy_grid)
                
                # Add shifted diagram lines
                if len(diagram_shifted) > 0:
                    # Create temporary dataframe with shifted energies for Lorentzian calculation
                    temp_diagram = diagram_lines.copy()
                    temp_diagram['energy'] = diagram_shifted['energy']
                    shifted_lorentzian += create_lorentzian_spectrum(temp_diagram, energy_grid)
                
                # Add shifted satellite lines
                if satellite_shifted is not None and len(satellite_shifted) > 0 and satellite_lines is not None:
                    # Create temporary dataframe with shifted energies for Lorentzian calculation
                    temp_satellite = satellite_lines.copy()
                    temp_satellite['energy'] = satellite_shifted['energy']
                    shifted_lorentzian += create_lorentzian_spectrum(temp_satellite, energy_grid)
                
                # Filter to current range
                mask = (energy_grid >= x_min) & (energy_grid <= x_max)
                if np.any(mask):
                    spectrum_filtered = shifted_lorentzian[mask]
                    energy_filtered = energy_grid[mask]
                else:
                    spectrum_filtered = shifted_lorentzian
                    energy_filtered = energy_grid
                
                fig.add_trace(go.Scatter(x=energy_filtered, 
                                         y=spectrum_filtered,
                                         mode='lines',
                                         name='Lorentzian Broadened',
                                         line=dict(color='purple', width=3),
                                         fill='tozeroy',
                                         fillcolor='rgba(128, 0, 128, 0.3)'),
                              row=2, col=1)
            
            # Update layout with increased height for spacing
            fig.update_layout(height=750,  # Increased height
                              showlegend=True,
                              template='plotly_white',
                              font=dict(size=12),
                              margin=dict(l=50, r=50, t=50, b=50))
            
            fig.update_xaxes(title_text='Energy [eV]', 
                             range=[x_min, x_max], 
                             tickformat='.0f',
                             row=1, col=1)
            fig.update_xaxes(title_text='Energy [eV]', 
                             range=[x_min, x_max], 
                             tickformat='.0f',
                             row=2, col=1)
            fig.update_yaxes(title_text='Intensity', row=1, col=1)
            fig.update_yaxes(title_text='Intensity', row=2, col=1)
            
            fig.show()
        
        # Update value display
        with value_output:
            value_output.clear_output(wait=True)
            print(f"Current Energy Shifts:")
            print(f"  Diagram lines: {diagram_shift:+.1f} eV")
            print(f"  Satellite lines: {satellite_shift:+.1f} eV")
            print(f"  X-axis range: {x_min:.0f} - {x_max:.0f} eV")
    
    # Connect sliders to update function
    diagram_slider.observe(lambda change: update_plot(diagram_slider.value, satellite_slider.value, x_min_slider.value, x_max_slider.value), names='value')
    satellite_slider.observe(lambda change: update_plot(diagram_slider.value, satellite_slider.value, x_min_slider.value, x_max_slider.value), names='value')
    x_min_slider.observe(lambda change: update_plot(diagram_slider.value, satellite_slider.value, x_min_slider.value, x_max_slider.value), names='value')
    x_max_slider.observe(lambda change: update_plot(diagram_slider.value, satellite_slider.value, x_min_slider.value, x_max_slider.value), names='value')
    
    # Initial plot
    update_plot(0, 0, energy_min, energy_max)
    
    # Return the interface with better layout
    controls_box = VBox([
        widgets.HTML('<h4 style="margin: 0; color: #666;">Energy Shifts</h4>'),
        HBox([diagram_slider, satellite_slider], layout=widgets.Layout(margin='5px 0')),
        widgets.HTML('<h4 style="margin: 10px 0 0 0; color: #666;">X-Axis Range</h4>'),
        HBox([x_min_slider, x_max_slider, reset_button], layout=widgets.Layout(margin='5px 0')),
        widgets.HTML('<h4 style="margin: 10px 0 0 0; color: #666;">Precise Values</h4>'),
        HBox([diagram_text, satellite_text], layout=widgets.Layout(margin='5px 0'))
    ], layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='10px 0'))
    
    return VBox([
        output,
        value_output,
        controls_box
    ], layout=widgets.Layout(width='100%'))
import os
import sys

from IPython.display import display
import ipywidgets as widgets

from .calculation import Sim5


sys.path.append(os.getcwd() + '/functions')
os.chdir(os.getcwd() + '/functions')  # Anpassung des Working Directories


def main():
    """
    Creates and starts sim5 widgets
    @return:
    """

    def reset_values(*args):
        """
        Reset widet parameters to initial values
        @return:
        """
        sigma_n.value = -40
        n_realisierungen.value = 3
        alpha_lms.value = 1
        N.value = 1500
        L.value = 549
        C.value = 2048

    def update_N(*args):
        """
        Adjust widget slider length for parameter N according to
        system impulse response length.

        @param args:
        @return:
        """
        if w_system.value == '700':
            N.max = 700
        elif w_system.value == '1500':
            N.max = 1500
        else:
            raise AssertionError(
                'Something went wrong with the system impulse response!')

    # Initalisieren der Widgets
    w_system = widgets.ToggleButtons(
        options=['1500', '700'],
        description='System Impulse Response Length',
        disabled=False,
        style={'description_width': 'initial'},
    )

    sigma_n = widgets.FloatSlider(
        description='$\u03C3_n^2$ / dB:',
        value=-40,
        min=-90,
        max=10,
        step=1,
        continuous_update=False,
        style={'description_width': 'initial'},
    )

    alpha_lms = widgets.FloatSlider(
        description='$ \u03B1_{LMS}:$',
        value=1,
        min=0,
        max=1,
        step=0.1,
        continuous_update=False,
        style={'description_width': 'initial'},
    )

    alpha_flms = widgets.FloatSlider(
        description='$\u03B1_{FLMS}$:',
        value=1,
        min=0,
        max=1,
        step=0.1,
        continuous_update=False,
        style={'description_width': 'initial'},
    )

    n_realisierungen = widgets.IntSlider(
        description='# Realizations:',
        value=3,
        min=1,
        max=50,
        step=1,
        continuous_update=False,
        style={'description_width': 'initial'},
    )

    N = widgets.IntSlider(
        description="N:",
        value=1500,
        min=0,
        max=1600,
        step=1,
        continuous_update=False,
        style={'description_width': 'initial'},
    )

    L = widgets.IntSlider(
        description="L:",
        value=549,
        min=0,
        max=550,
        step=1,
        continuous_update=False,
        style={'description_width': 'initial'},
    )

    C = widgets.IntSlider(
        description="C:",
        value=2048,
        min=0,
        max=2048,
        step=1,
        continuous_update=False,
        style={'description_width': 'initial'},
    )

    # Zurücksetzen der Werte auf Startwerte
    reset_button = widgets.Button(description="Reset")
    reset_button.on_click(reset_values)

    # Intslider auf Länge der Impulsantwort anpassen
    w_system.observe(update_N, 'value')

    # Widgets anordnen
    box_1 = widgets.HBox([sigma_n, n_realisierungen, alpha_lms, alpha_flms])
    box_2 = widgets.HBox([N, L, C])
    box_3 = widgets.HBox([reset_button, w_system])
    box = widgets.VBox([box_1, box_2])

    interactive_plot = widgets.interactive(
        sim,
        sigma_n=sigma_n,
        n_realisierungen=n_realisierungen,
        alpha_lms=alpha_lms,
        alpha_flms=alpha_flms,
        N=N,
        L=L,
        C=C,
        w_system=w_system,
    )

    # Show Plots
    display(box_3)
    display(box)
    display(interactive_plot.children[-1])


def sim(alpha_flms, alpha_lms, C, L, N, n_realisierungen, sigma_n, w_system):
    sim_5 = Sim5(alpha_flms, alpha_lms, C, L, N, n_realisierungen,
                 sigma_n, w_system)
    sim_5.calc_filter_coefficients()
    sim_5.show_results()


if __name__ == "__main__":
    main()

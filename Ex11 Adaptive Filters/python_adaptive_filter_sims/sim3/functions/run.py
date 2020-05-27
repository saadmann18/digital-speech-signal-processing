from .s3a import *
from .s3b import *

import ipywidgets as widgets
from IPython.display import display


def run_s3a():
    """
    Start der Simulation S3a
    @return:
    """
    def reset_values(*args):
        ac.value = 90
        sigma.value = -30
        loops.value = 20
        mu.value = 0.05

    ac = widgets.FloatSlider(
        description='Cosine Power percentage $a_c$',
        value=90, min=0, max=99.99, step=1,
        continuous_update=False,
        style={'description_width': 'initial'}
    )
    sigma = widgets.FloatSlider(
        description='$\u03C3_n^2$ / dB',
        value=-30, min=-90, max=10, step=1,
        continuous_update=False,
        style={'description_width': 'initial'}
    )
    loops = widgets.IntSlider(
        description='# Realizations:',
        value=20, min=1, max=50, step=1,
        continuous_update=False,
        style={'description_width': 'initial'}
    )
    mu = widgets.FloatSlider(
        description="\u03BC:",
        value=0.05, min=0, max=1, step=0.05,
        continuous_update=False,
        style={'description_width': 'initial'}
    )

    reset_button = widgets.Button(description="Reset")
    reset_button.on_click(reset_values)

    interactive_plot = widgets.interactive(s3a_main, ac=ac, sigma=sigma,
                                           loops=loops, mu=mu)
    display(reset_button)
    display(interactive_plot)


def s3a_main(ac, sigma, loops, mu):
    s3a = S3a(ac, sigma, loops, mu)
    s3a.calc_filter_coefficients()
    s3a.plot_results()


def run_s3b():
    """
    Start der Simulation S3b
    @return:
    """
    def reset_values(*args):
        ac.value = 0
        sigma.value = -30
        loops.value = 16
        mu.value = '0.05'
        p5.value = '2   4   8  16 '

    ac = widgets.FloatSlider(
        description='Cosine power percentage $a_c$',
        value=0, min=0, max=99.99, step=1,
        continuous_update=False,
        style={'description_width': 'initial'}
    )

    sigma = widgets.FloatSlider(
        description='$\u03C3_n^2$ / dB',
        value=-30, min=-90, max=10, step=1,
        continuous_update=False,
        style={'description_width': 'initial'}
    )
    loops = widgets.IntSlider(
        description="# Realizations:",
        value=16, min=1, max=50, step=1,
        continuous_update=False,
        style={'description_width': 'initial'}
    )

    mu = widgets.Text(
        value='0.05',
        placeholder='Input list of all $\mu$ values',
        description='$\mu$:',
        disabled=False,
        continuous_update=False
    )

    p5 = widgets.Text(
        value='2   4   8  16 ',
        placeholder='Input list of all N values',
        description='N :',
        disabled=False,
        continuous_update=False
    )

    reset_button = widgets.Button(description="Reset")
    reset_button.on_click(reset_values)

    interactive_plot = widgets.interactive(s3b_main, ac=ac, sigma=sigma,
                                           loops=loops, mu=mu, p5=p5)
    display(reset_button)
    display(interactive_plot)


def s3b_main(ac, sigma, loops, mu, p5):
    s3b = S3b(ac, sigma, loops, mu, p5)
    s3b.calc_filter_coefficients()
    s3b.plot_results()

from matplotlib import colors as mcolors

THEME_COLORS = [
    "#80cdff",  # blue
    "#ffca80",  # orange
    "#60e37a",  # green
    "#ff80b1",  # pink
    "#bd80ff",  # purple
    "#000000",  # black
]


def line_color(i: int) -> str:
    """
    Compute a color for the i-th line in a plot. Only works for up to 6 lines.

    :param i: The index of the line.
    :returns: The color for the line.
    """
    return modify_color(THEME_COLORS[i], 0.5, 0.9)


def bar_color(i: int) -> str:
    """
    Compute a color for the i-th bar in a plot. Only works for up to 6 bars.
    Somewhat lighter than the line color.

    :param i: The index of the bar.
    :returns: The color for the bar.
    """
    return modify_color(THEME_COLORS[i], 1.0, 1.0)


def modify_color(color, saturation_change, value_change):
    m = mcolors.ColorConverter().to_rgb
    rgb = m(color)
    hsv = mcolors.rgb_to_hsv(rgb)
    hsv[1] = 1 - (1 - hsv[1]) * saturation_change
    hsv[2] *= value_change
    color = mcolors.hsv_to_rgb(hsv)
    return color

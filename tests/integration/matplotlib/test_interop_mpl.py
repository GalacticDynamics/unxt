# pylint: disable=import-error, too-many-lines

"""Test the Array API."""

import pytest

from unxt import Quantity

plt = pytest.importorskip("matplotlib.pyplot")


@pytest.mark.mpl_image_compare
def test_labels_axes() -> plt.Figure:
    fig, ax = plt.subplots()
    x = Quantity([1, 2, 3], "kpc")
    y = Quantity([1, 4, 9], "Msun")
    ax.plot(x, y)
    return fig

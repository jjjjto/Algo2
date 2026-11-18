import pytest

tk = pytest.importorskip("tkinter")

from seam_carving.gui import SeamCarvingApp


def test_gui_builds_without_loading_an_image():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    app = SeamCarvingApp(root)
    assert app.method.get() == "dp"
    assert app.current is None
    root.update()
    root.destroy()

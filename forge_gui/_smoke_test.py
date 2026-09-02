"""Headless smoke test — construction + page switching + refresh (no render)."""
import sys
from PySide6.QtWidgets import QApplication
from forge_gui.app import MainWindow, _NUM_PAGES


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1320, 860)
    assert w.pages.count() == _NUM_PAGES, \
        f"expected {_NUM_PAGES} pages, got {w.pages.count()}"
    for i in range(w.pages.count()):
        w.sidebar._select(i)
        app.processEvents()
        pg = w.pages.widget(i)
        if hasattr(pg, "refresh"):
            pg.refresh()
        app.processEvents()
        print(f"page {i} ({w.page_title.text()}): OK")
    for _ in range(3):
        w._refresh_fast()
        w._refresh_slow()
        app.processEvents()
    print(f"ALL OK — {w.pages.count()} pages constructed, switched, refreshed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

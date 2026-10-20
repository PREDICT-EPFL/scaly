# /// script
# dependencies = ["playwright"]
# ///
"""Render the published benchmark pages and check the October 20 lead annotations in Firefox."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main(base_url: str, out: Path) -> None:
  out.mkdir(parents=True, exist_ok=True)
  results, errors = [], []
  with sync_playwright() as browser_api:
    browser = browser_api.firefox.launch(headless=True, executable_path=str(Path.home() / ".cache/ms-playwright/firefox-1543/firefox/firefox"))
    page = browser.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1)
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    for path, count in (("", 4), ("scalability/", 7), ("closed_loop/", 3)):
      page.goto(f"{base_url}/benchmarks/{path}", wait_until="networkidle")
      page.wait_for_function("document.querySelectorAll('.vega-chart svg').length === " + str(count), timeout=60000)
      for mode in ("default", "slate"):
        page.evaluate("mode => document.body.setAttribute('data-md-color-scheme', mode)", mode)
        page.wait_for_function("document.querySelectorAll('.vega-chart svg').length === " + str(count), timeout=60000)
        page.wait_for_timeout(1200)
        figures = page.locator(".vega-chart")
        assert figures.count() == count
        for index in range(count):
          figure = figures.nth(index)
          spec = figure.get_attribute("data-spec")
          text = figure.locator("svg").text_content()
          if spec.endswith("sweep_runtime.vl.json"):
            assert all(label in text for label in ("7.1× faster", "18× faster", "19× faster", "8.7× faster")), text
          figure.screenshot(path=str(out / f"{Path(spec).stem}-{mode}.png"))
          results.append(dict(page=path, mode=mode, spec=spec, labels=text))
    browser.close()
  (out / "verification.json").write_text(json.dumps(dict(figures=results, errors=errors), indent=2) + "\n")
  assert not errors, errors
  print(f"Rendered {len(results)} figures across three pages and two color schemes without chart errors.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--url", default="http://127.0.0.1:8781")
  parser.add_argument("--out", required=True, type=Path)
  args = parser.parse_args()
  main(args.url, args.out.resolve())

// Renders each <div class="vega-chart" data-spec="…"> with the shared benchmark style, and again
// when the reader switches between light and dark mode. Specs and config carry the light colors;
// DARK maps each one to its dark-mode step.
const DARK = {
  "#1baf7a": "#199e70", "#eb6834": "#d95926", "#2a78d6": "#3987e5", "#eda100": "#c98500",
  "#e87ba4": "#d55181", "#4a3aa7": "#9085e9", "#e34948": "#e66767",
  "#0b0b0b": "#e8e8e6", "#52514e": "#c3c2b7", "#e1e0d9": "#2a2d35", "#c3c2b7": "#4a4d57", "#ffffff": "#0b0c0f",
}

const load = url => fetch(url).then(response => response.json())

function darken(value) {
  return JSON.parse(JSON.stringify(value).replace(/#[0-9a-f]{6}/g, hex => DARK[hex] ?? hex))
}

async function render(element) {
  const url = new URL(element.dataset.spec, location.href)
  const [spec, config] = await Promise.all([load(url), load(new URL("config.json", url))])
  const dark = document.body.getAttribute("data-md-color-scheme") === "slate"
  config.font = getComputedStyle(element).fontFamily
  await vegaEmbed(element, dark ? darken(spec) : spec, {
    config: dark ? darken(config) : config,
    renderer: "svg",
    actions: false,
    loader: vega.loader({ baseURL: new URL(".", url).href }),
  })
}

const renderAll = () => document.querySelectorAll(".vega-chart").forEach(render)

document$.subscribe(renderAll)
new MutationObserver(renderAll).observe(document.body, { attributeFilter: ["data-md-color-scheme"] })

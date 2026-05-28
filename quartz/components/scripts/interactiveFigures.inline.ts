import {
  axisBottom,
  axisLeft,
  bin as d3Bin,
  curveMonotoneX,
  line as d3Line,
  max,
  range,
  scaleLinear,
  select,
} from "d3"

const gaussianFigureSelector = '[data-interactive-figure="gaussian-sample-histogram"]'
const gaussianSampleStops = [10, 100, 1000, 10000]
const interactiveLightboxId = "interactive-figure-lightbox"

type GaussianState = {
  binCount: number
  mu: number
  n: number
  seed: number
  sigma: number
  samples: number[]
}

type GaussianController = {
  destroy: () => void
  figure: HTMLElement
  render: () => void
  state: GaussianState
  syncFrom: (state: GaussianState) => void
}

type GaussianFigureOptions = {
  expanded?: boolean
  onExpand?: (controller: GaussianController) => void
  useGlobalCleanup?: boolean
}

let interactiveFigureCount = 0
const gaussianControllers = new WeakMap<HTMLElement, GaussianController>()
const lightboxCleanups = new WeakMap<HTMLElement, () => void>()
const lightboxSources = new WeakMap<HTMLElement, HTMLElement>()

function readNumber(value: string | undefined, fallback: number) {
  if (!value) return fallback

  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function closestSampleStop(value: number) {
  return gaussianSampleStops.reduce((closest, current) =>
    Math.abs(current - value) < Math.abs(closest - value) ? current : closest,
  )
}

function sampleStopIndex(value: number) {
  return gaussianSampleStops.indexOf(closestSampleStop(value))
}

function seededRandom(seed: number) {
  let state = seed >>> 0

  return () => {
    state = (1664525 * state + 1013904223) >>> 0
    return state / 4294967296
  }
}

function sampleExtent(values: number[]) {
  let min = Infinity
  let max = -Infinity

  for (const value of values) {
    if (value < min) min = value
    if (value > max) max = value
  }

  return [min, max] as const
}

function makeGaussianSamples(mu: number, sigma: number, n: number, seed: number) {
  const random = seededRandom(seed)
  const samples: number[] = []

  while (samples.length < n) {
    const u = Math.max(random(), Number.EPSILON)
    const v = random()
    const radius = Math.sqrt(-2 * Math.log(u))
    const angle = 2 * Math.PI * v

    samples.push(mu + sigma * radius * Math.cos(angle))
    if (samples.length < n) {
      samples.push(mu + sigma * radius * Math.sin(angle))
    }
  }

  return samples
}

function mean(values: number[]) {
  return values.reduce((total, value) => total + value, 0) / values.length
}

function standardDeviation(values: number[], center: number) {
  const variance = values.reduce((total, value) => total + (value - center) ** 2, 0) / values.length
  return Math.sqrt(variance)
}

function normalPdf(x: number, mu: number, sigma: number) {
  const z = (x - mu) / sigma
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI))
}

function cssColor(element: HTMLElement, name: string, fallback: string) {
  const color = getComputedStyle(element).getPropertyValue(name).trim()
  return color || fallback
}

function setTooltipPosition(figure: HTMLElement, tooltip: HTMLElement, event: PointerEvent) {
  const bounds = figure.getBoundingClientRect()
  const x = event.clientX - bounds.left
  const y = event.clientY - bounds.top

  tooltip.style.left = `${x}px`
  tooltip.style.top = `${y}px`
}

function renderGaussianHistogram(figure: HTMLElement, state: GaussianState) {
  const plot = figure.querySelector<HTMLElement>(".gaussian-histogram-plot")
  const stats = figure.querySelector<HTMLElement>(".gaussian-histogram-stats")
  const tooltip = figure.querySelector<HTMLElement>(".gaussian-histogram-tooltip")
  const output = figure.querySelector<HTMLOutputElement>(".gaussian-histogram-n")
  if (!plot || !stats || !tooltip || !output) return

  const width = Math.max(plot.clientWidth, 224)
  const isExpanded = figure.classList.contains("is-expanded")
  const height = isExpanded ? Math.min(Math.max(width * 0.48, 340), 460) : width < 250 ? 174 : 190
  const margin = isExpanded
    ? { top: 22, right: 22, bottom: 42, left: 50 }
    : { top: 14, right: 12, bottom: 28, left: 34 }
  const innerWidth = width - margin.left - margin.right
  const innerHeight = height - margin.top - margin.bottom
  const sampleMean = mean(state.samples)
  const sampleStd = standardDeviation(state.samples, sampleMean)
  const [sampleMin, sampleMax] = sampleExtent(state.samples)
  const domain: [number, number] = [
    Math.min(state.mu - 4 * state.sigma, sampleMin),
    Math.max(state.mu + 4 * state.sigma, sampleMax),
  ]

  const xScale = scaleLinear().domain(domain).range([0, innerWidth]).nice()
  const histogram = d3Bin<number, number>()
    .domain(xScale.domain() as [number, number])
    .thresholds(xScale.ticks(state.binCount))
    .value((d) => d)

  const bins = histogram(state.samples)
  const curveX = range(0, 96).map((i) => domain[0] + (i * (domain[1] - domain[0])) / 95)
  const trueDensity = curveX.map(
    (x) => [x, normalPdf(x, state.mu, state.sigma)] as [number, number],
  )
  const fittedDensity = curveX.map(
    (x) => [x, normalPdf(x, sampleMean, sampleStd)] as [number, number],
  )
  const barMax = max(bins, (d) => {
    const width = (d.x1 ?? 0) - (d.x0 ?? 0)
    return width > 0 ? d.length / (state.samples.length * width) : 0
  })
  const curveMax = max([...trueDensity, ...fittedDensity], (d) => d[1])
  const yScale = scaleLinear()
    .domain([0, Math.max(barMax ?? 0, curveMax ?? 0) * 1.16])
    .range([innerHeight, 0])
    .nice()

  const histColor = cssColor(figure, "--chart-hist", "#8dbfe8")
  const histStroke = cssColor(figure, "--chart-hist-stroke", "#2c5f9e")
  const trueColor = cssColor(figure, "--chart-true", "#9d3b42")
  const fitColor = cssColor(figure, "--chart-fit", "#2c7d54")
  const axisColor = cssColor(figure, "--chart-axis", "#6c645a")
  const gridColor = cssColor(figure, "--chart-grid", "#e3ddd2")

  plot.replaceChildren()
  output.value = String(state.n)
  figure.dataset.n = String(state.n)
  figure.dataset.seed = String(state.seed)
  stats.textContent = `n=${state.n}  mean=${sampleMean.toFixed(2)}  std=${sampleStd.toFixed(2)}`

  const svg = select(plot)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img")
    .attr("aria-label", "Gaussian sample histogram with true and empirical density curves")

  const chart = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`)

  chart
    .append("g")
    .attr("class", "gaussian-histogram-grid")
    .call(
      axisLeft(yScale)
        .ticks(4)
        .tickSize(-innerWidth)
        .tickFormat(() => ""),
    )
    .call((g) => g.select(".domain").remove())
    .call((g) => g.selectAll("line").attr("stroke", gridColor))

  chart
    .append("g")
    .attr("class", "gaussian-histogram-bars")
    .selectAll("rect")
    .data(bins)
    .join("rect")
    .attr("x", (d) => xScale(d.x0 ?? 0) + 1)
    .attr("y", (d) => {
      const width = (d.x1 ?? 0) - (d.x0 ?? 0)
      const density = width > 0 ? d.length / (state.samples.length * width) : 0
      return yScale(density)
    })
    .attr("width", (d) => Math.max(0, xScale(d.x1 ?? 0) - xScale(d.x0 ?? 0) - 1))
    .attr("height", (d) => {
      const width = (d.x1 ?? 0) - (d.x0 ?? 0)
      const density = width > 0 ? d.length / (state.samples.length * width) : 0
      return innerHeight - yScale(density)
    })
    .attr("fill", histColor)
    .attr("stroke", histStroke)
    .attr("stroke-width", 0.7)
    .attr("rx", 1.4)
    .on("pointerenter", (event: PointerEvent, d) => {
      const width = (d.x1 ?? 0) - (d.x0 ?? 0)
      const density = width > 0 ? d.length / (state.samples.length * width) : 0

      tooltip.innerHTML = `<strong>${(d.x0 ?? 0).toFixed(2)} to ${(d.x1 ?? 0).toFixed(
        2,
      )}</strong><span>count=${d.length}, density=${density.toFixed(3)}</span>`
      tooltip.hidden = false
      setTooltipPosition(figure, tooltip, event)
    })
    .on("pointermove", (event: PointerEvent) => {
      setTooltipPosition(figure, tooltip, event)
    })
    .on("pointerleave", () => {
      tooltip.hidden = true
    })

  const densityLine = d3Line<[number, number]>()
    .x((d) => xScale(d[0]))
    .y((d) => yScale(d[1]))
    .curve(curveMonotoneX)

  chart
    .append("path")
    .datum(trueDensity)
    .attr("class", "gaussian-histogram-line")
    .attr("d", densityLine)
    .attr("fill", "none")
    .attr("stroke", trueColor)
    .attr("stroke-linecap", "round")
    .attr("stroke-width", 2.2)

  chart
    .append("path")
    .datum(fittedDensity)
    .attr("class", "gaussian-histogram-line")
    .attr("d", densityLine)
    .attr("fill", "none")
    .attr("stroke", fitColor)
    .attr("stroke-dasharray", "4 3")
    .attr("stroke-linecap", "round")
    .attr("stroke-width", 1.9)

  chart
    .append("g")
    .attr("class", "gaussian-histogram-axis")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(
      axisBottom(xScale)
        .ticks(isExpanded ? 8 : width < 260 ? 4 : 5)
        .tickSizeOuter(0),
    )

  chart
    .append("g")
    .attr("class", "gaussian-histogram-axis")
    .call(
      axisLeft(yScale)
        .ticks(isExpanded ? 6 : 4)
        .tickSizeOuter(0),
    )

  svg
    .selectAll(".gaussian-histogram-axis path, .gaussian-histogram-axis line")
    .attr("stroke", axisColor)
  svg.selectAll(".gaussian-histogram-axis text").attr("fill", axisColor)
}

function initialGaussianState(figure: HTMLElement): GaussianState {
  const n = closestSampleStop(Math.round(readNumber(figure.dataset.n, 1000)))
  const state = {
    binCount: Math.round(readNumber(figure.dataset.bins, 30)),
    mu: readNumber(figure.dataset.mu, 2.5),
    n,
    seed: Math.round(readNumber(figure.dataset.seed, 42)),
    sigma: readNumber(figure.dataset.sigma, 1.2),
    samples: [] as number[],
  }

  state.samples = makeGaussianSamples(state.mu, state.sigma, state.n, state.seed)
  return state
}

function cloneGaussianState(state: GaussianState): GaussianState {
  return {
    ...state,
    samples: [...state.samples],
  }
}

function createGaussianFigure(
  figure: HTMLElement,
  options: GaussianFigureOptions = {},
): GaussianController {
  const id = `gaussian-histogram-${++interactiveFigureCount}`
  const disposers: (() => void)[] = []
  const state = initialGaussianState(figure)
  let controller: GaussianController

  figure.classList.add("interactive-figure", "gaussian-histogram-figure")
  figure.classList.toggle("is-expanded", options.expanded === true)
  figure.classList.toggle("can-expand", options.onExpand !== undefined)
  figure.replaceChildren()
  figure.innerHTML = `
    <div class="interactive-figure-toolbar">
      <div class="interactive-figure-title">Gaussian samples</div>
      <div class="interactive-figure-actions">
        <button class="interactive-figure-resample" type="button" aria-label="Draw new samples">
          <svg aria-hidden="true" viewBox="0 0 24 24">
            <path d="M20 12a8 8 0 1 1-2.34-5.66" />
            <path d="M20 4v6h-6" />
          </svg>
        </button>
        ${
          options.onExpand
            ? `<button class="interactive-figure-expand" type="button" aria-label="Open interactive figure">
          <svg aria-hidden="true" viewBox="0 0 24 24">
            <path d="M8 3H5a2 2 0 0 0-2 2v3" />
            <path d="M16 3h3a2 2 0 0 1 2 2v3" />
            <path d="M8 21H5a2 2 0 0 1-2-2v-3" />
            <path d="M16 21h3a2 2 0 0 0 2-2v-3" />
          </svg>
        </button>`
            : ""
        }
      </div>
    </div>
    <div class="gaussian-histogram-plot"></div>
    <div class="gaussian-histogram-legend" aria-hidden="true">
      <span><i class="is-true"></i>True</span>
      <span><i class="is-fit"></i>Fit</span>
    </div>
    <div class="gaussian-histogram-stats"></div>
    <label class="gaussian-histogram-slider">
      <div class="gaussian-histogram-slider-row">
        <span>n</span>
        <input type="range" min="0" max="${gaussianSampleStops.length - 1}" step="1" value="${sampleStopIndex(
          state.n,
        )}" aria-describedby="${id}-ticks" />
        <output class="gaussian-histogram-n">${state.n}</output>
      </div>
      <div class="gaussian-histogram-ticks" id="${id}-ticks" aria-hidden="true">
        ${gaussianSampleStops.map((stop) => `<span>${stop}</span>`).join("")}
      </div>
    </label>
    <div class="gaussian-histogram-tooltip" hidden></div>
  `

  const resampleButton = figure.querySelector<HTMLButtonElement>(".interactive-figure-resample")
  const expandButton = figure.querySelector<HTMLButtonElement>(".interactive-figure-expand")
  const slider = figure.querySelector<HTMLInputElement>(".gaussian-histogram-slider input")
  const plot = figure.querySelector<HTMLElement>(".gaussian-histogram-plot")
  const output = figure.querySelector<HTMLOutputElement>(".gaussian-histogram-n")
  const syncControls = () => {
    if (slider) slider.value = String(sampleStopIndex(state.n))
    if (output) output.value = String(state.n)
  }
  const render = () => {
    syncControls()
    renderGaussianHistogram(figure, state)
  }
  const resample = () => {
    state.seed += 1
    state.samples = makeGaussianSamples(state.mu, state.sigma, state.n, state.seed)
    render()
  }
  const updateSampleSize = () => {
    if (!slider) return

    state.n = gaussianSampleStops[Number(slider.value)] ?? 1000
    state.samples = makeGaussianSamples(state.mu, state.sigma, state.n, state.seed)
    render()
  }
  const expand = () => options.onExpand?.(controller)
  const expandFromFigure = (event: MouseEvent) => {
    const target = event.target
    if (!(target instanceof Element)) return
    if (target.closest("button, input, label, a, .gaussian-histogram-tooltip")) return

    expand()
  }

  resampleButton?.addEventListener("click", resample)
  expandButton?.addEventListener("click", expand)
  if (options.onExpand) figure.addEventListener("click", expandFromFigure)
  slider?.addEventListener("input", updateSampleSize)
  disposers.push(() => {
    resampleButton?.removeEventListener("click", resample)
    expandButton?.removeEventListener("click", expand)
    if (options.onExpand) figure.removeEventListener("click", expandFromFigure)
    slider?.removeEventListener("input", updateSampleSize)
  })

  const observer = new ResizeObserver(render)
  if (plot) observer.observe(plot)
  disposers.push(() => observer.disconnect())

  controller = {
    destroy: () => {
      for (const dispose of disposers) dispose()
      gaussianControllers.delete(figure)
    },
    figure,
    render,
    state,
    syncFrom: (nextState) => {
      state.binCount = nextState.binCount
      state.mu = nextState.mu
      state.n = nextState.n
      state.seed = nextState.seed
      state.sigma = nextState.sigma
      state.samples = [...nextState.samples]
      render()
    },
  }
  gaussianControllers.set(figure, controller)
  requestAnimationFrame(render)

  if (options.useGlobalCleanup !== false) {
    window.addCleanup(controller.destroy)
  }

  return controller
}

function closeInteractiveLightbox(immediate = false) {
  const lightbox = document.getElementById(interactiveLightboxId)
  if (!lightbox) return

  const modalFigure = lightbox.querySelector<HTMLElement>(gaussianFigureSelector)
  const sourceFigure = lightboxSources.get(lightbox)
  const sourceController = sourceFigure ? gaussianControllers.get(sourceFigure) : undefined
  const modalController = modalFigure ? gaussianControllers.get(modalFigure) : undefined

  if (sourceController && modalController) {
    sourceController.syncFrom(modalController.state)
  }

  modalController?.destroy()
  lightboxCleanups.get(lightbox)?.()
  lightboxCleanups.delete(lightbox)
  lightboxSources.delete(lightbox)
  lightbox.classList.remove("is-open")
  document.body.classList.remove("interactive-figure-lightbox-open")
  if (immediate) {
    lightbox.remove()
  } else {
    window.setTimeout(() => lightbox.remove(), 180)
  }
}

function openGaussianLightbox(sourceController: GaussianController) {
  closeInteractiveLightbox(true)

  const lightbox = document.createElement("div")
  lightbox.id = interactiveLightboxId
  lightbox.className = "interactive-figure-lightbox"
  lightbox.setAttribute("aria-hidden", "true")
  lightboxSources.set(lightbox, sourceController.figure)
  lightbox.innerHTML = `
    <button class="interactive-figure-lightbox-close" type="button" aria-label="Close interactive figure">
      <svg aria-hidden="true" viewBox="0 0 24 24">
        <path d="M18 6 6 18M6 6l12 12" />
      </svg>
    </button>
    <div class="interactive-figure-lightbox-frame" role="dialog" aria-modal="true">
      <figure
        class="side-figure interactive-figure"
        data-interactive-figure="gaussian-sample-histogram"
        data-mu="${sourceController.state.mu}"
        data-sigma="${sourceController.state.sigma}"
        data-n="${sourceController.state.n}"
        data-bins="${sourceController.state.binCount}"
        data-seed="${sourceController.state.seed}"
      ></figure>
    </div>
  `

  const closeButton = lightbox.querySelector<HTMLButtonElement>(
    ".interactive-figure-lightbox-close",
  )
  const modalFigure = lightbox.querySelector<HTMLElement>(gaussianFigureSelector)
  const closeFromBackdrop = (event: MouseEvent) => {
    if (event.target === lightbox) closeInteractiveLightbox()
  }
  const closeFromButton = () => closeInteractiveLightbox()
  const closeFromKeyboard = (event: KeyboardEvent) => {
    if (event.key === "Escape") closeInteractiveLightbox()
  }

  lightbox.addEventListener("click", closeFromBackdrop)
  closeButton?.addEventListener("click", closeFromButton)
  document.addEventListener("keydown", closeFromKeyboard)
  lightboxCleanups.set(lightbox, () => {
    lightbox.removeEventListener("click", closeFromBackdrop)
    closeButton?.removeEventListener("click", closeFromButton)
    document.removeEventListener("keydown", closeFromKeyboard)
  })
  document.body.append(lightbox)
  document.body.classList.add("interactive-figure-lightbox-open")

  if (modalFigure) {
    const modalController = createGaussianFigure(modalFigure, {
      expanded: true,
      useGlobalCleanup: false,
    })
    modalController.syncFrom(cloneGaussianState(sourceController.state))
  }

  requestAnimationFrame(() => {
    lightbox.classList.add("is-open")
    lightbox.setAttribute("aria-hidden", "false")
    closeButton?.focus()
  })
}

function setupInteractiveFigures() {
  const figures = document.querySelectorAll<HTMLElement>(`article ${gaussianFigureSelector}`)

  for (const figure of figures) {
    if (figure.dataset.interactiveReady === "true") continue

    figure.dataset.interactiveReady = "true"
    createGaussianFigure(figure, {
      onExpand: openGaussianLightbox,
    })
  }
}

document.addEventListener("nav", setupInteractiveFigures)

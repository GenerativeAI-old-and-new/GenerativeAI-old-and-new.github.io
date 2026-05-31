import {
  axisBottom,
  axisLeft,
  contours as d3Contours,
  geoPath,
  interpolateViridis,
  line as d3Line,
  max,
  range,
  scaleLinear,
  select,
  symbol,
  symbolStar,
} from "d3"
import { InteractiveFigureDefinition } from "./core"
import { readNumber, seededRandom } from "./math"
import { expandButtonHtml } from "./ui"

type OptimizerCase = "gd-vs-sgd" | "gd-vs-momentum" | "signed-vs-softsign" | "adam-comparison"
type MethodKey = "gd" | "sgd" | "momentum" | "sign" | "softsign" | "adam"

type Point = {
  x: number
  y: number
}

type TrajectoryPoint = Point & {
  loss: number
}

type Trajectory = {
  method: MethodKey
  points: TrajectoryPoint[]
}

type OptimizerCaseConfig = {
  betaDefault: number
  domainX: [number, number]
  domainY: [number, number]
  etaDefault: number
  etaRange: [number, number]
  epsilonDefault: number
  kx: number
  ky: number
  methods: MethodKey[]
  minimum: Point
  noiseDefault: number
  start: Point
  steps: number
  title: string
}

type OptimizerState = {
  adaptivity: number
  beta: number
  caseKey: OptimizerCase
  epsilon: number
  eta: number
  frame: number
  noise: number
  seed: number
  startX: number
  startY: number
  steps: number
}

const optimizerCases: Record<OptimizerCase, OptimizerCaseConfig> = {
  "gd-vs-sgd": {
    betaDefault: 0.8,
    domainX: [-5.4, 6.6],
    domainY: [-6.6, 6.6],
    etaDefault: 0.45,
    etaRange: [0.08, 0.85],
    epsilonDefault: 0.12,
    kx: 0.13,
    ky: 0.95,
    methods: ["gd", "sgd"],
    minimum: { x: 2, y: -3 },
    noiseDefault: 0.55,
    start: { x: -4, y: 4 },
    steps: 34,
    title: "GD vs noisy SGD",
  },
  "gd-vs-momentum": {
    betaDefault: 0.76,
    domainX: [-5.3, 5.3],
    domainY: [-5.3, 5.3],
    etaDefault: 0.158,
    etaRange: [0.06, 0.176],
    epsilonDefault: 0.12,
    kx: 0.16,
    ky: 11.2,
    methods: ["gd", "momentum"],
    minimum: { x: 0, y: 0 },
    noiseDefault: 0,
    start: { x: 2, y: 2 },
    steps: 34,
    title: "GD vs Momentum",
  },
  "signed-vs-softsign": {
    betaDefault: 0.82,
    domainX: [-2.8, 2.75],
    domainY: [-2.25, 2.25],
    etaDefault: 0.1,
    etaRange: [0.04, 0.16],
    epsilonDefault: 0.14,
    kx: 0.16,
    ky: 1.65,
    methods: ["gd", "sign", "softsign"],
    minimum: { x: 0, y: 0 },
    noiseDefault: 0,
    start: { x: 2, y: 1 },
    steps: 42,
    title: "GD, Sign, SoftSign",
  },
  "adam-comparison": {
    betaDefault: 0.86,
    domainX: [-2.8, 2.75],
    domainY: [-2.25, 2.25],
    etaDefault: 0.1,
    etaRange: [0.04, 0.16],
    epsilonDefault: 0.12,
    kx: 0.16,
    ky: 1.65,
    methods: ["gd", "momentum", "sign", "softsign", "adam"],
    minimum: { x: 0, y: 0 },
    noiseDefault: 0,
    start: { x: 2, y: 1 },
    steps: 42,
    title: "Adam in context",
  },
}

const caseKeys = Object.keys(optimizerCases) as OptimizerCase[]
const methodLabels: Record<MethodKey, string> = {
  adam: "Adam",
  gd: "GD",
  momentum: "Momentum",
  sgd: "SGD",
  sign: "Sign",
  softsign: "SoftSign",
}
const methodOffsets: Record<MethodKey, number> = {
  adam: 941,
  gd: 17,
  momentum: 313,
  sgd: 109,
  sign: 577,
  softsign: 733,
}
const sliderStep = 0.001
const contourGrid = {
  x: 150,
  y: 110,
}
let markerCount = 0

function isOptimizerCase(value: string | undefined): value is OptimizerCase {
  return caseKeys.includes(value as OptimizerCase)
}

function clamp(value: number, minValue: number, maxValue: number) {
  return Math.min(maxValue, Math.max(minValue, value))
}

function clampPoint(point: Point, config: OptimizerCaseConfig): Point {
  return {
    x: clamp(point.x, config.domainX[0], config.domainX[1]),
    y: clamp(point.y, config.domainY[0], config.domainY[1]),
  }
}

function formatNumber(value: number, digits = 2) {
  return Number.parseFloat(value.toFixed(digits)).toString()
}

function contourColor(index: number, count: number) {
  const ratio = count <= 1 ? 0 : index / (count - 1)
  return interpolateViridis(0.08 + 0.86 * clamp(ratio, 0, 1))
}

function loss(point: Point, config: OptimizerCaseConfig) {
  const dx = point.x - config.minimum.x
  const dy = point.y - config.minimum.y
  return 0.5 * (config.kx * dx * dx + config.ky * dy * dy)
}

function gradient(point: Point, config: OptimizerCaseConfig): Point {
  return {
    x: config.kx * (point.x - config.minimum.x),
    y: config.ky * (point.y - config.minimum.y),
  }
}

function sign(value: number) {
  if (Math.abs(value) < 1e-9) return 0
  return value < 0 ? -1 : 1
}

function gaussianRandom(random: () => number) {
  const u = Math.max(random(), Number.EPSILON)
  const v = random()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function readOptimizerState(figure: HTMLElement): OptimizerState {
  const caseKey = isOptimizerCase(figure.dataset.case) ? figure.dataset.case : "gd-vs-sgd"
  const config = optimizerCases[caseKey]
  const eta = clamp(readNumber(figure.dataset.eta, config.etaDefault), ...config.etaRange)
  const steps = Math.round(clamp(readNumber(figure.dataset.steps, config.steps), 8, 80))
  const startX = clamp(readNumber(figure.dataset.startX, config.start.x), ...config.domainX)
  const startY = clamp(readNumber(figure.dataset.startY, config.start.y), ...config.domainY)

  return {
    adaptivity: clamp(readNumber(figure.dataset.adaptivity, 0.78), 0, 1),
    beta: clamp(readNumber(figure.dataset.beta, config.betaDefault), 0, 0.96),
    caseKey,
    epsilon: clamp(readNumber(figure.dataset.epsilon, config.epsilonDefault), 0, 0.5),
    eta,
    frame: Math.round(clamp(readNumber(figure.dataset.frame, steps), 0, steps)),
    noise: clamp(readNumber(figure.dataset.noise, config.noiseDefault), 0, 1.2),
    seed: Math.round(readNumber(figure.dataset.seed, 37)),
    startX,
    startY,
    steps,
  }
}

function cloneOptimizerState(state: OptimizerState): OptimizerState {
  return { ...state }
}

function resetStart(state: OptimizerState) {
  const config = optimizerCases[state.caseKey]
  state.startX = config.start.x
  state.startY = config.start.y
  state.frame = state.steps
}

function simulateMethod(
  method: MethodKey,
  state: OptimizerState,
  config: OptimizerCaseConfig,
): Trajectory {
  const random = seededRandom(state.seed + methodOffsets[method])
  const points: TrajectoryPoint[] = []
  let point = clampPoint({ x: state.startX, y: state.startY }, config)
  let m = { x: 0, y: 0 }
  let v = { x: 0, y: 0 }
  const adamBeta1 = 0.55 + 0.35 * state.adaptivity
  const adamBeta2 = 0.7 + 0.28 * state.adaptivity

  points.push({ ...point, loss: loss(point, config) })

  for (let step = 0; step < state.steps; step++) {
    const g = gradient(point, config)
    let update = { ...g }

    if (method === "sgd") {
      update = {
        x: g.x + state.noise * 0.52 * gaussianRandom(random),
        y: g.y + state.noise * 0.52 * gaussianRandom(random),
      }
    } else if (method === "momentum") {
      m = {
        x: state.beta * m.x + (1 - state.beta) * g.x,
        y: state.beta * m.y + (1 - state.beta) * g.y,
      }
      // Keep the learning-rate knob comparable with GD while the displayed path shows smoothing.
      update = {
        x: m.x / Math.max(1 - state.beta, 0.04),
        y: m.y / Math.max(1 - state.beta, 0.04),
      }
    } else if (method === "sign") {
      update = {
        x: sign(g.x),
        y: sign(g.y),
      }
    } else if (method === "softsign") {
      update = {
        x: g.x / (Math.abs(g.x) + state.epsilon),
        y: g.y / (Math.abs(g.y) + state.epsilon),
      }
    } else if (method === "adam") {
      m = {
        x: adamBeta1 * m.x + (1 - adamBeta1) * g.x,
        y: adamBeta1 * m.y + (1 - adamBeta1) * g.y,
      }
      v = {
        x: adamBeta2 * v.x + (1 - adamBeta2) * g.x * g.x,
        y: adamBeta2 * v.y + (1 - adamBeta2) * g.y * g.y,
      }
      update = {
        x: m.x / (Math.sqrt(v.x) + 0.04),
        y: m.y / (Math.sqrt(v.y) + 0.04),
      }
    }

    point = clampPoint(
      {
        x: point.x - state.eta * update.x,
        y: point.y - state.eta * update.y,
      },
      config,
    )
    points.push({ ...point, loss: loss(point, config) })
  }

  return { method, points }
}

function simulateTrajectories(state: OptimizerState) {
  const config = optimizerCases[state.caseKey]
  return config.methods.map((method) => simulateMethod(method, state, config))
}

function optimizerToolbarHtml(title: string, canExpand: boolean) {
  return `<div class="interactive-figure-toolbar optimizer-trajectory-toolbar">
    <div class="interactive-figure-title">${title}</div>
    <div class="interactive-figure-actions">
      <button class="interactive-figure-resample" type="button" aria-label="Reset optimizer path">
        <svg aria-hidden="true" viewBox="0 0 24 24">
          <path d="M20 12a8 8 0 1 1-2.34-5.66" />
          <path d="M20 4v6h-6" />
        </svg>
      </button>
      ${canExpand ? expandButtonHtml() : ""}
    </div>
  </div>`
}

function legendHtml(methods: MethodKey[]) {
  return methods
    .map(
      (method) =>
        `<span class="optimizer-legend-item optimizer-method-${method}"><i></i>${methodLabels[method]}</span>`,
    )
    .join("")
}

function sliderHtml(
  className: string,
  label: string,
  value: number,
  minValue: number,
  maxValue: number,
  step = sliderStep,
) {
  return `<label>
    <span>${label}</span>
    <input class="${className}" type="range" min="${minValue}" max="${maxValue}" step="${step}" value="${value}" />
    <output></output>
  </label>`
}

function numberInputHtml(
  className: string,
  label: string,
  value: number,
  minValue: number,
  maxValue: number,
) {
  return `<label>
    <span>${label}</span>
    <input class="${className}" type="number" min="${minValue}" max="${maxValue}" step="0.05" value="${formatNumber(
      value,
      2,
    )}" />
  </label>`
}

function startControlsHtml(state: OptimizerState, config: OptimizerCaseConfig) {
  return `<div class="optimizer-start-controls">
    ${numberInputHtml("optimizer-start-x-input", "x0", state.startX, config.domainX[0], config.domainX[1])}
    ${numberInputHtml("optimizer-start-y-input", "y0", state.startY, config.domainY[0], config.domainY[1])}
  </div>`
}

function caseSpecificControls(state: OptimizerState) {
  if (state.caseKey === "gd-vs-sgd") {
    return sliderHtml("optimizer-noise", "noise", state.noise, 0, 1.2, 0.01)
  }

  if (state.caseKey === "gd-vs-momentum") {
    return sliderHtml("optimizer-beta", "beta", state.beta, 0, 0.96, 0.01)
  }

  if (state.caseKey === "signed-vs-softsign") {
    return sliderHtml("optimizer-epsilon", "epsilon", state.epsilon, 0.01, 0.5, 0.01)
  }

  return sliderHtml("optimizer-adaptivity", "adapt", state.adaptivity, 0, 1, 0.01)
}

function optimizerTemplate({ canExpand, state }: { canExpand: boolean; state: OptimizerState }) {
  const config = optimizerCases[state.caseKey]

  return `
    ${optimizerToolbarHtml(config.title, canExpand)}
    <div class="interactive-figure-plot optimizer-trajectory-plot"></div>
    <div class="optimizer-trajectory-footer" data-no-expand>
      <div class="optimizer-trajectory-legend" aria-hidden="true">
        ${legendHtml(config.methods)}
      </div>
      <div class="optimizer-trajectory-readout" aria-live="polite"></div>
      ${startControlsHtml(state, config)}
      <div class="optimizer-trajectory-controls">
        ${sliderHtml("optimizer-eta", "eta", state.eta, config.etaRange[0], config.etaRange[1])}
        ${caseSpecificControls(state)}
        ${sliderHtml("optimizer-frame", "step", state.frame, 0, state.steps, 1)}
      </div>
    </div>
  `
}

function updateOutput(input: HTMLInputElement | null, value: string) {
  const output = input?.parentElement?.querySelector("output")
  if (output) output.textContent = value
}

function syncOptimizerControls(figure: HTMLElement, state: OptimizerState) {
  const config = optimizerCases[state.caseKey]
  const eta = figure.querySelector<HTMLInputElement>(".optimizer-eta")
  const noise = figure.querySelector<HTMLInputElement>(".optimizer-noise")
  const beta = figure.querySelector<HTMLInputElement>(".optimizer-beta")
  const epsilon = figure.querySelector<HTMLInputElement>(".optimizer-epsilon")
  const adaptivity = figure.querySelector<HTMLInputElement>(".optimizer-adaptivity")
  const frame = figure.querySelector<HTMLInputElement>(".optimizer-frame")
  const startX = figure.querySelector<HTMLInputElement>(".optimizer-start-x-input")
  const startY = figure.querySelector<HTMLInputElement>(".optimizer-start-y-input")

  if (eta) eta.value = String(state.eta)
  if (noise) noise.value = String(state.noise)
  if (beta) beta.value = String(state.beta)
  if (epsilon) epsilon.value = String(state.epsilon)
  if (adaptivity) adaptivity.value = String(state.adaptivity)
  if (startX) startX.value = formatNumber(state.startX, 2)
  if (startY) startY.value = formatNumber(state.startY, 2)
  if (frame) {
    frame.max = String(state.steps)
    frame.value = String(state.frame)
  }

  updateOutput(eta, formatNumber(state.eta, 3))
  updateOutput(noise, formatNumber(state.noise, 2))
  updateOutput(beta, formatNumber(state.beta, 2))
  updateOutput(epsilon, formatNumber(state.epsilon, 2))
  updateOutput(adaptivity, formatNumber(state.adaptivity, 2))
  updateOutput(frame, `${state.frame}/${state.steps}`)

  figure.dataset.case = state.caseKey
  figure.dataset.eta = String(state.eta)
  figure.dataset.frame = String(state.frame)
  figure.dataset.noise = String(state.noise)
  figure.dataset.beta = String(state.beta)
  figure.dataset.epsilon = String(state.epsilon)
  figure.dataset.adaptivity = String(state.adaptivity)
  figure.dataset.seed = String(state.seed)
  figure.dataset.startX = String(state.startX)
  figure.dataset.startY = String(state.startY)
  figure.dataset.steps = String(config.steps)
}

function contourValues(config: OptimizerCaseConfig) {
  const values: number[] = []
  for (let row = 0; row < contourGrid.y; row++) {
    const y =
      config.domainY[1] - (row / (contourGrid.y - 1)) * (config.domainY[1] - config.domainY[0])

    for (let col = 0; col < contourGrid.x; col++) {
      const x =
        config.domainX[0] + (col / (contourGrid.x - 1)) * (config.domainX[1] - config.domainX[0])
      values.push(loss({ x, y }, config))
    }
  }

  return values
}

function renderOptimizer(figure: HTMLElement, state: OptimizerState) {
  const plot = figure.querySelector<HTMLElement>(".optimizer-trajectory-plot")
  if (!plot) return

  const config = optimizerCases[state.caseKey]
  const width = Math.max(plot.clientWidth, 320)
  const isExpanded = figure.classList.contains("is-expanded")
  const isCompact = width < 620 && !isExpanded
  const margin = {
    bottom: isCompact ? 38 : 42,
    left: isCompact ? 42 : 52,
    right: isCompact ? 18 : 30,
    top: isCompact ? 22 : 28,
  }
  const lossGap = isExpanded ? 36 : isCompact ? 38 : 46
  const lossHeight = isExpanded ? 88 : isCompact ? 70 : 86
  const modalHeightBudget = isExpanded ? clamp(window.innerHeight - 320, 440, 660) : undefined
  const mainHeight = isExpanded
    ? Math.max(280, (modalHeightBudget ?? 660) - margin.top - margin.bottom - lossGap - lossHeight)
    : isCompact
      ? 278
      : 366
  const height = margin.top + mainHeight + lossGap + lossHeight + margin.bottom
  const innerWidth = width - margin.left - margin.right
  const lossTop = margin.top + mainHeight + lossGap
  const markerId = `optimizer-arrow-${++markerCount}`
  const trajectories = simulateTrajectories(state)
  const visibleFrame = clamp(state.frame, 0, state.steps)
  const xScale = scaleLinear().domain(config.domainX).range([0, innerWidth])
  const yScale = scaleLinear().domain(config.domainY).range([mainHeight, 0])
  const lossScaleX = scaleLinear().domain([0, state.steps]).range([0, innerWidth])
  const maxLoss =
    max(trajectories.flatMap((trajectory) => trajectory.points.map((point) => point.loss))) ?? 1
  const lossScaleY = scaleLinear()
    .domain([0, maxLoss * 1.05])
    .range([lossHeight, 0])
  const pathLine = d3Line<TrajectoryPoint>()
    .x((point) => xScale(point.x))
    .y((point) => yScale(point.y))
  const lossLine = d3Line<TrajectoryPoint>()
    .x((_, index) => lossScaleX(index))
    .y((point) => lossScaleY(point.loss))
  const values = contourValues(config)
  const maxValue = max(values) ?? 1
  const thresholds = range(1, 19).map((index) => maxValue * (index / 19) ** 1.7)
  const contourData = d3Contours().size([contourGrid.x, contourGrid.y]).thresholds(thresholds)(
    values,
  )
  const contourLineData = contourData
  const contourPath = geoPath()
  const readout = figure.querySelector<HTMLElement>(".optimizer-trajectory-readout")

  if (readout) {
    const best = trajectories
      .map((trajectory) => {
        const point = trajectory.points[Math.min(visibleFrame, trajectory.points.length - 1)]
        return `${methodLabels[trajectory.method]} ${formatNumber(point.loss, 3)}`
      })
      .join(" · ")
    readout.textContent = `loss at step ${visibleFrame}: ${best}`
  }

  plot.replaceChildren()

  const svg = select(plot)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img")
    .attr("aria-label", `${config.title} optimizer trajectories on a quadratic loss contour`)

  const defs = svg.append("defs")
  for (const method of config.methods) {
    defs
      .append("marker")
      .attr("id", `${markerId}-${method}`)
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 8)
      .attr("refY", 5)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("class", `optimizer-arrow-marker optimizer-method-${method}`)
      .attr("d", "M0,0 L10,5 L0,10 Z")
  }

  const main = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`)
  const lossPanel = svg.append("g").attr("transform", `translate(${margin.left},${lossTop})`)

  main
    .append("rect")
    .attr("class", "optimizer-drag-surface")
    .attr("width", innerWidth)
    .attr("height", mainHeight)
    .attr("rx", 4)
    .style("--contour-color", contourColor(0, contourData.length))

  const contourGroup = main
    .append("g")
    .attr(
      "transform",
      `scale(${innerWidth / (contourGrid.x - 1)}, ${mainHeight / (contourGrid.y - 1)})`,
    )

  contourGroup
    .selectAll("path.optimizer-contour-fill")
    .data(contourData)
    .join("path")
    .attr("class", "optimizer-contour-fill")
    .attr("d", contourPath)
    .style("--contour-color", (_, index) => contourColor(index, contourData.length))

  contourGroup
    .selectAll("path.optimizer-contour-line")
    .data(contourLineData)
    .join("path")
    .attr("class", "optimizer-contour-line")
    .attr("d", contourPath)
    .style("--contour-color", (_, index) => contourColor(index, contourLineData.length))

  main
    .append("g")
    .attr("class", "optimizer-grid")
    .attr("transform", `translate(0,${mainHeight})`)
    .call(
      axisBottom(xScale)
        .ticks(isCompact ? 4 : 6)
        .tickSize(-mainHeight)
        .tickFormat(() => ""),
    )

  main
    .append("g")
    .attr("class", "optimizer-grid")
    .call(
      axisLeft(yScale)
        .ticks(isCompact ? 4 : 6)
        .tickSize(-innerWidth)
        .tickFormat(() => ""),
    )

  main
    .append("g")
    .attr("class", "optimizer-axis")
    .attr("transform", `translate(0,${mainHeight})`)
    .call(axisBottom(xScale).ticks(isCompact ? 4 : 6))

  main
    .append("g")
    .attr("class", "optimizer-axis")
    .call(axisLeft(yScale).ticks(isCompact ? 4 : 6))

  main
    .append("text")
    .attr("class", "optimizer-axis-label")
    .attr("x", innerWidth / 2)
    .attr("y", mainHeight + 32)
    .attr("text-anchor", "middle")
    .text("x")

  main
    .append("text")
    .attr("class", "optimizer-axis-label")
    .attr("transform", "rotate(-90)")
    .attr("x", -mainHeight / 2)
    .attr("y", -34)
    .attr("text-anchor", "middle")
    .text("y")

  const trajectoryGroup = main.append("g").attr("class", "optimizer-trajectories")

  for (const trajectory of trajectories) {
    const visiblePoints = trajectory.points.slice(0, visibleFrame + 1)
    const methodClass = `optimizer-method-${trajectory.method}`

    trajectoryGroup
      .append("path")
      .datum(visiblePoints)
      .attr("class", `optimizer-path ${methodClass}`)
      .attr("d", pathLine)

    trajectoryGroup
      .selectAll(`circle.${methodClass}`)
      .data(visiblePoints.filter((_, index) => index % (isCompact ? 3 : 2) === 0))
      .join("circle")
      .attr("class", `optimizer-step ${methodClass}`)
      .attr("cx", (point) => xScale(point.x))
      .attr("cy", (point) => yScale(point.y))
      .attr("r", isCompact ? 2.4 : 3)

    if (visiblePoints.length > 1) {
      const current = visiblePoints[visiblePoints.length - 1]
      const previous = visiblePoints[visiblePoints.length - 2]
      if (Math.hypot(current.x - previous.x, current.y - previous.y) > 0.01) {
        trajectoryGroup
          .append("line")
          .attr("class", `optimizer-update-arrow ${methodClass}`)
          .attr("x1", xScale(previous.x))
          .attr("y1", yScale(previous.y))
          .attr("x2", xScale(current.x))
          .attr("y2", yScale(current.y))
          .attr("marker-end", `url(#${markerId}-${trajectory.method})`)
      }
    }

    const finalPoint = visiblePoints[visiblePoints.length - 1]
    trajectoryGroup
      .append("circle")
      .attr("class", `optimizer-final ${methodClass}`)
      .attr("cx", xScale(finalPoint.x))
      .attr("cy", yScale(finalPoint.y))
      .attr("r", isCompact ? 3.6 : 4.4)
  }

  const starPath = symbol()
    .type(symbolStar)
    .size(isCompact ? 110 : 150)()
  main
    .append("path")
    .attr("class", "optimizer-minimum")
    .attr("transform", `translate(${xScale(config.minimum.x)},${yScale(config.minimum.y)})`)
    .attr("d", starPath)

  main
    .append("circle")
    .attr("class", "optimizer-start")
    .attr("cx", xScale(state.startX))
    .attr("cy", yScale(state.startY))
    .attr("r", isCompact ? 5.2 : 6.2)

  const dragSurface = main.select<SVGRectElement>(".optimizer-drag-surface")
  const startHit = main
    .append("circle")
    .attr("class", "optimizer-start-hit")
    .attr("cx", xScale(state.startX))
    .attr("cy", yScale(state.startY))
    .attr("r", isCompact ? 16 : 18)

  const updateStartFromEvent = (event: PointerEvent) => {
    const bounds = plot.getBoundingClientRect()
    if (bounds.width === 0 || bounds.height === 0) return

    const viewBoxX = ((event.clientX - bounds.left) / bounds.width) * width
    const viewBoxY = ((event.clientY - bounds.top) / bounds.height) * height
    state.startX = clamp(xScale.invert(viewBoxX - margin.left), ...config.domainX)
    state.startY = clamp(yScale.invert(viewBoxY - margin.top), ...config.domainY)
    state.frame = state.steps
    syncOptimizerControls(figure, state)
    renderOptimizer(figure, state)
  }
  const beginDrag = (event: PointerEvent) => {
    event.preventDefault()
    updateStartFromEvent(event)

    const move = (moveEvent: PointerEvent) => updateStartFromEvent(moveEvent)
    const end = () => {
      window.removeEventListener("pointermove", move)
      window.removeEventListener("pointerup", end)
      window.removeEventListener("pointercancel", end)
    }

    window.addEventListener("pointermove", move)
    window.addEventListener("pointerup", end)
    window.addEventListener("pointercancel", end)
  }

  dragSurface.on("pointerdown", beginDrag)
  startHit.on("pointerdown", beginDrag)

  lossPanel
    .append("rect")
    .attr("class", "optimizer-loss-background")
    .attr("width", innerWidth)
    .attr("height", lossHeight)
    .attr("rx", 4)

  lossPanel
    .append("g")
    .attr("class", "optimizer-grid")
    .attr("transform", `translate(0,${lossHeight})`)
    .call(
      axisBottom(lossScaleX)
        .ticks(isCompact ? 4 : 6)
        .tickSize(-lossHeight)
        .tickFormat(() => ""),
    )

  lossPanel
    .append("g")
    .attr("class", "optimizer-loss-axis")
    .attr("transform", `translate(0,${lossHeight})`)
    .call(axisBottom(lossScaleX).ticks(isCompact ? 4 : 6))

  lossPanel
    .append("g")
    .attr("class", "optimizer-loss-axis")
    .call(axisLeft(lossScaleY).ticks(isCompact ? 2 : 3))

  lossPanel
    .append("text")
    .attr("class", "optimizer-panel-label")
    .attr("x", 0)
    .attr("y", -12)
    .text("loss")

  for (const trajectory of trajectories) {
    const visiblePoints = trajectory.points.slice(0, visibleFrame + 1)
    const methodClass = `optimizer-method-${trajectory.method}`

    lossPanel
      .append("path")
      .datum(visiblePoints)
      .attr("class", `optimizer-loss-line ${methodClass}`)
      .attr("d", lossLine)
  }
}

function bindOptimizerControls({
  addCleanup,
  figure,
  render,
  state,
}: {
  addCleanup: (cleanup: () => void) => void
  figure: HTMLElement
  render: () => void
  state: OptimizerState
}) {
  const bindNumberInput = (
    selector: string,
    update: (value: number) => void,
    options: { integer?: boolean } = {},
  ) => {
    const input = figure.querySelector<HTMLInputElement>(selector)
    const onInput = () => {
      if (!input) return

      if (input.value.trim() === "") return
      const value = options.integer ? Math.round(Number(input.value)) : Number(input.value)
      if (!Number.isFinite(value)) return
      update(value)
      render()
    }

    input?.addEventListener("input", onInput)
    addCleanup(() => input?.removeEventListener("input", onInput))
  }

  bindNumberInput(".optimizer-eta", (value) => {
    const config = optimizerCases[state.caseKey]
    state.eta = clamp(value, ...config.etaRange)
  })
  bindNumberInput(".optimizer-noise", (value) => {
    state.noise = clamp(value, 0, 1.2)
  })
  bindNumberInput(".optimizer-beta", (value) => {
    state.beta = clamp(value, 0, 0.96)
  })
  bindNumberInput(".optimizer-epsilon", (value) => {
    state.epsilon = clamp(value, 0.01, 0.5)
  })
  bindNumberInput(".optimizer-adaptivity", (value) => {
    state.adaptivity = clamp(value, 0, 1)
  })
  bindNumberInput(".optimizer-start-x-input", (value) => {
    const config = optimizerCases[state.caseKey]
    state.startX = clamp(value, ...config.domainX)
    state.frame = state.steps
  })
  bindNumberInput(".optimizer-start-y-input", (value) => {
    const config = optimizerCases[state.caseKey]
    state.startY = clamp(value, ...config.domainY)
    state.frame = state.steps
  })
  bindNumberInput(
    ".optimizer-frame",
    (value) => {
      state.frame = Math.round(clamp(value, 0, state.steps))
    },
    { integer: true },
  )
}

export const optimizerTrajectoryFigure: InteractiveFigureDefinition<OptimizerState> = {
  bindControls: bindOptimizerControls,
  classNames: ["optimizer-trajectory-figure"],
  cloneState: cloneOptimizerState,
  expandIgnoreSelector:
    "button, input, label, a, select, [data-no-expand], .optimizer-drag-surface, .optimizer-start-hit",
  readState: readOptimizerState,
  render: renderOptimizer,
  resample: (state) => {
    if (state.caseKey === "gd-vs-sgd") {
      state.seed += 1
    } else {
      resetStart(state)
    }
  },
  serializeState: (state) => ({
    adaptivity: state.adaptivity,
    beta: state.beta,
    case: state.caseKey,
    epsilon: state.epsilon,
    eta: state.eta,
    frame: state.frame,
    noise: state.noise,
    seed: state.seed,
    "start-x": state.startX,
    "start-y": state.startY,
    steps: state.steps,
  }),
  syncControls: syncOptimizerControls,
  template: optimizerTemplate,
  type: "optimizer-trajectory",
}

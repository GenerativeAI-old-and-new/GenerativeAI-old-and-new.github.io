import {
  area as d3Area,
  axisBottom,
  axisLeft,
  curveMonotoneX,
  line as d3Line,
  range,
  scaleLinear,
  select,
} from "d3"
import { InteractiveFigureDefinition } from "./core"
import { readNumber } from "./math"
import { expandButtonHtml } from "./ui"

type ActivationKey = "relu" | "tanh" | "sigmoid" | "softplus" | "leakyRelu"
type TargetKey = "wave" | "cusp" | "bump" | "multiwave" | "bumps" | "chirp"

type UniversalApproximationState = {
  activationKey: ActivationKey
  probeX: number
  targetKey: TargetKey
  width: number
}

type TargetDefinition = {
  fn: (x: number) => number
  key: TargetKey
  label: string
}

type ActivationDefinition = {
  fn: (z: number) => number
  key: ActivationKey
  label: string
  scale: number
}

type ApproximationModel = {
  centers: number[]
  linearWeights: number[]
  networkWeights: number[]
}

type CurvePoint = {
  error: number
  linear: number
  network: number
  target: number
  x: number
}

type UnitPoint = {
  value: number
  x: number
}

const domain: [number, number] = [-1, 1]
const widthDomain: [number, number] = [1, 32]
const defaultState: UniversalApproximationState = {
  activationKey: "relu",
  probeX: 0.35,
  targetKey: "wave",
  width: 8,
}
const curvePointCount = 260
const fitPointCount = 181
const lensPointCount = 96
const ridge = 1e-4

const targetKeys: TargetKey[] = ["wave", "cusp", "bump", "multiwave", "bumps", "chirp"]
const activationKeys: ActivationKey[] = ["relu", "tanh", "sigmoid", "softplus", "leakyRelu"]

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x))
}

function softplus(x: number) {
  if (x > 30) return x
  if (x < -30) return Math.exp(x)

  return Math.log1p(Math.exp(x))
}

const targetDefinitions: Record<TargetKey, TargetDefinition> = {
  bump: {
    fn: (x) => 0.78 * (sigmoid(13 * (x + 0.44)) - sigmoid(13 * (x - 0.38))) - 0.25,
    key: "bump",
    label: "smooth step/bump",
  },
  bumps: {
    fn: (x) =>
      0.76 * Math.exp(-46 * (x + 0.52) ** 2) -
      0.5 * Math.exp(-70 * (x - 0.02) ** 2) +
      0.58 * Math.exp(-38 * (x - 0.48) ** 2) -
      0.12,
    key: "bumps",
    label: "localized bumps",
  },
  chirp: {
    fn: (x) =>
      0.42 * Math.sin(Math.PI * (2.2 * x + 3.8 * x * x)) + 0.2 * Math.sin(8 * Math.PI * x * x),
    key: "chirp",
    label: "chirp",
  },
  cusp: {
    fn: (x) => 0.74 * Math.abs(x + 0.08) - 0.34 + 0.1 * Math.sin(3 * Math.PI * x),
    key: "cusp",
    label: "cusp",
  },
  multiwave: {
    fn: (x) =>
      0.34 * Math.sin(3 * Math.PI * x) +
      0.2 * Math.sin(7 * Math.PI * (x + 0.12)) +
      0.13 * Math.cos(11 * Math.PI * x),
    key: "multiwave",
    label: "multi-frequency wave",
  },
  wave: {
    fn: (x) => 0.48 * Math.sin(2 * Math.PI * (x + 0.08)) + 0.18 * Math.sin(5 * Math.PI * x),
    key: "wave",
    label: "smooth wave",
  },
}

const activationDefinitions: Record<ActivationKey, ActivationDefinition> = {
  leakyRelu: {
    fn: (z) => (z >= 0 ? z : 0.12 * z),
    key: "leakyRelu",
    label: "leaky ReLU",
    scale: 4.6,
  },
  relu: {
    fn: (z) => Math.max(0, z),
    key: "relu",
    label: "ReLU",
    scale: 4.6,
  },
  sigmoid: {
    fn: sigmoid,
    key: "sigmoid",
    label: "sigmoid",
    scale: 5.4,
  },
  softplus: {
    fn: softplus,
    key: "softplus",
    label: "softplus",
    scale: 5,
  },
  tanh: {
    fn: Math.tanh,
    key: "tanh",
    label: "tanh",
    scale: 4.2,
  },
}

function clamp(value: number, minValue: number, maxValue: number) {
  return Math.min(maxValue, Math.max(minValue, value))
}

function clampWidth(value: number) {
  return Math.round(clamp(value, widthDomain[0], widthDomain[1]))
}

function formatNumber(value: number, digits = 3) {
  if (!Number.isFinite(value)) return "inf"

  return Number.parseFloat(value.toFixed(digits)).toString()
}

function readActivationKey(value: string | undefined): ActivationKey {
  return activationKeys.includes(value as ActivationKey) ? (value as ActivationKey) : "relu"
}

function readTargetKey(value: string | undefined): TargetKey {
  return targetKeys.includes(value as TargetKey) ? (value as TargetKey) : "wave"
}

function grid(count: number) {
  return range(0, count).map((index) => domain[0] + ((domain[1] - domain[0]) * index) / (count - 1))
}

function hiddenCenters(width: number) {
  if (width <= 1) return [0]

  return range(0, width).map((index) => -0.95 + (1.9 * index) / (width - 1))
}

function hiddenFeature(x: number, center: number, activationKey: ActivationKey) {
  const activation = activationDefinitions[activationKey]

  return activation.fn(activation.scale * (x - center))
}

function networkFeatureRow(x: number, centers: number[], activationKey: ActivationKey) {
  return [1, x, ...centers.map((center) => hiddenFeature(x, center, activationKey))]
}

function predict(weights: number[], features: number[]) {
  return weights.reduce((total, weight, index) => total + weight * features[index], 0)
}

function solveLinearSystem(matrix: number[][], rhs: number[]) {
  const n = rhs.length
  const augmented = matrix.map((row, index) => [...row, rhs[index]])

  for (let column = 0; column < n; column++) {
    let pivotRow = column
    let pivotAbs = Math.abs(augmented[column][column])

    for (let row = column + 1; row < n; row++) {
      const candidate = Math.abs(augmented[row][column])
      if (candidate > pivotAbs) {
        pivotAbs = candidate
        pivotRow = row
      }
    }

    if (pivotAbs < 1e-12) continue
    if (pivotRow !== column) {
      const current = augmented[column]
      augmented[column] = augmented[pivotRow]
      augmented[pivotRow] = current
    }

    const pivot = augmented[column][column]
    for (let row = column + 1; row < n; row++) {
      const factor = augmented[row][column] / pivot
      if (factor === 0) continue

      for (let col = column; col <= n; col++) {
        augmented[row][col] -= factor * augmented[column][col]
      }
    }
  }

  const solution = new Array<number>(n).fill(0)
  for (let row = n - 1; row >= 0; row--) {
    let total = augmented[row][n]
    for (let col = row + 1; col < n; col++) {
      total -= augmented[row][col] * solution[col]
    }

    const pivot = augmented[row][row]
    solution[row] = Math.abs(pivot) < 1e-12 ? 0 : total / pivot
  }

  return solution
}

function fitWeights(features: number[][], values: number[], ridgeStrength: number) {
  const featureCount = features[0]?.length ?? 0
  const lhs = range(0, featureCount).map(() => new Array<number>(featureCount).fill(0))
  const rhs = new Array<number>(featureCount).fill(0)

  for (let rowIndex = 0; rowIndex < features.length; rowIndex++) {
    const row = features[rowIndex]
    const y = values[rowIndex]

    for (let i = 0; i < featureCount; i++) {
      rhs[i] += row[i] * y
      for (let j = 0; j < featureCount; j++) {
        lhs[i][j] += row[i] * row[j]
      }
    }
  }

  for (let i = 0; i < featureCount; i++) {
    lhs[i][i] += i < 2 ? ridgeStrength * 0.1 : ridgeStrength
  }

  return solveLinearSystem(lhs, rhs)
}

function buildModel(state: UniversalApproximationState): ApproximationModel {
  const target = targetDefinitions[state.targetKey]
  const xs = grid(fitPointCount)
  const values = xs.map(target.fn)
  const centers = hiddenCenters(state.width)
  const linearFeatures = xs.map((x) => [1, x])
  const networkFeatures = xs.map((x) => networkFeatureRow(x, centers, state.activationKey))

  return {
    centers,
    linearWeights: fitWeights(linearFeatures, values, ridge),
    networkWeights: fitWeights(networkFeatures, values, ridge),
  }
}

function makeCurvePoints(state: UniversalApproximationState, model: ApproximationModel) {
  const target = targetDefinitions[state.targetKey]

  return grid(curvePointCount).map((x) => {
    const targetValue = target.fn(x)
    const linear = predict(model.linearWeights, [1, x])
    const network = predict(
      model.networkWeights,
      networkFeatureRow(x, model.centers, state.activationKey),
    )

    return {
      error: Math.abs(targetValue - network),
      linear,
      network,
      target: targetValue,
      x,
    }
  })
}

function yExtent(points: CurvePoint[]) {
  let minValue = Infinity
  let maxValue = -Infinity

  for (const point of points) {
    minValue = Math.min(minValue, point.target, point.linear, point.network)
    maxValue = Math.max(maxValue, point.target, point.linear, point.network)
  }

  const padding = Math.max(0.08, (maxValue - minValue) * 0.12)
  return [minValue - padding, maxValue + padding] as [number, number]
}

function modelPrediction(x: number, state: UniversalApproximationState, model: ApproximationModel) {
  const target = targetDefinitions[state.targetKey].fn(x)
  const linear = predict(model.linearWeights, [1, x])
  const network = predict(
    model.networkWeights,
    networkFeatureRow(x, model.centers, state.activationKey),
  )

  return {
    error: Math.abs(target - network),
    linear,
    network,
    target,
    x,
  }
}

function readUniversalApproximationState(figure: HTMLElement): UniversalApproximationState {
  return {
    activationKey: readActivationKey(figure.dataset.activation),
    probeX: clamp(readNumber(figure.dataset.probe, defaultState.probeX), domain[0], domain[1]),
    targetKey: readTargetKey(figure.dataset.target),
    width: clampWidth(readNumber(figure.dataset.width, defaultState.width)),
  }
}

function cloneUniversalApproximationState(
  state: UniversalApproximationState,
): UniversalApproximationState {
  return { ...state }
}

function resetUniversalApproximation(state: UniversalApproximationState) {
  Object.assign(state, defaultState)
}

function universalApproximationToolbarHtml(canExpand: boolean) {
  return `<div class="interactive-figure-toolbar universal-approximation-toolbar">
    <div class="interactive-figure-title">From linear fit to neural approximation</div>
    <div class="interactive-figure-actions">
      <button class="interactive-figure-resample" type="button" aria-label="Reset approximation">
        <svg aria-hidden="true" viewBox="0 0 24 24">
          <path d="M20 12a8 8 0 1 1-2.34-5.66" />
          <path d="M20 4v6h-6" />
        </svg>
      </button>
      ${canExpand ? expandButtonHtml() : ""}
    </div>
  </div>`
}

function selectOptions<T extends string>(
  keys: T[],
  selected: T,
  definitions: Record<T, { label: string }>,
) {
  return keys
    .map((key) => {
      const selectedAttribute = key === selected ? " selected" : ""
      return `<option value="${key}"${selectedAttribute}>${definitions[key].label}</option>`
    })
    .join("")
}

function universalApproximationTemplate({
  canExpand,
  id,
  state,
}: {
  canExpand: boolean
  id: string
  state: UniversalApproximationState
}) {
  return `
    ${universalApproximationToolbarHtml(canExpand)}
    <div class="interactive-figure-plot universal-approximation-plot"></div>
    <div class="universal-approximation-footer" data-no-expand>
      <div class="universal-approximation-legend" aria-hidden="true">
        <span class="universal-approximation-legend-item universal-approximation-legend-target"><i></i>f*</span>
        <span class="universal-approximation-legend-item universal-approximation-legend-linear"><i></i>linear</span>
        <span class="universal-approximation-legend-item universal-approximation-legend-network"><i></i>network</span>
      </div>
      <div class="universal-approximation-readout" aria-live="polite"></div>
      <div class="universal-approximation-controls">
        <label class="universal-approximation-select-control">
          <span>target</span>
          <select class="universal-approximation-target" aria-label="Target function">
            ${selectOptions(targetKeys, state.targetKey, targetDefinitions)}
          </select>
        </label>
        <label class="universal-approximation-select-control">
          <span>activation</span>
          <select class="universal-approximation-activation" aria-label="Activation function">
            ${selectOptions(activationKeys, state.activationKey, activationDefinitions)}
          </select>
        </label>
        <label class="universal-approximation-width-control">
          <span>hidden units</span>
          <input class="universal-approximation-width" type="range" min="${widthDomain[0]}" max="${widthDomain[1]}" step="1" value="${state.width}" aria-describedby="${id}-width" />
          <output class="universal-approximation-width-output" id="${id}-width">${state.width}</output>
        </label>
      </div>
    </div>
  `
}

function syncUniversalApproximationControls(
  figure: HTMLElement,
  state: UniversalApproximationState,
) {
  const targetSelect = figure.querySelector<HTMLSelectElement>(".universal-approximation-target")
  const activationSelect = figure.querySelector<HTMLSelectElement>(
    ".universal-approximation-activation",
  )
  const widthSlider = figure.querySelector<HTMLInputElement>(".universal-approximation-width")
  const widthOutput = figure.querySelector<HTMLOutputElement>(
    ".universal-approximation-width-output",
  )

  if (targetSelect) targetSelect.value = state.targetKey
  if (activationSelect) activationSelect.value = state.activationKey
  if (widthSlider) widthSlider.value = String(state.width)
  if (widthOutput) widthOutput.value = String(state.width)

  figure.dataset.activation = state.activationKey
  figure.dataset.probe = String(state.probeX)
  figure.dataset.target = state.targetKey
  figure.dataset.width = String(state.width)
}

function updateReadout(
  figure: HTMLElement,
  state: UniversalApproximationState,
  points: CurvePoint[],
  probe: CurvePoint,
) {
  const readout = figure.querySelector<HTMLElement>(".universal-approximation-readout")
  if (!readout) return

  const meanError = points.reduce((total, point) => total + point.error, 0) / points.length
  const maxError = points.reduce((largest, point) => Math.max(largest, point.error), 0)

  readout.innerHTML = `
    <span>N ${state.width}</span>
    <span>mean err ${formatNumber(meanError)}</span>
    <span>max err ${formatNumber(maxError)}</span>
    <span>x ${formatNumber(probe.x, 2)} err ${formatNumber(probe.error)}</span>
  `
}

function renderUniversalApproximation(figure: HTMLElement, state: UniversalApproximationState) {
  const plot = figure.querySelector<HTMLElement>(".universal-approximation-plot")
  if (!plot) return

  const width = Math.max(plot.clientWidth, 320)
  const isExpanded = figure.classList.contains("is-expanded")
  const isCompact = width < 650 && !isExpanded
  const margin = {
    bottom: isCompact ? 34 : 40,
    left: isCompact ? 42 : 52,
    right: isCompact ? 18 : 30,
    top: isCompact ? 24 : 28,
  }
  const modalHeightBudget = isExpanded ? clamp(window.innerHeight - 320, 430, 650) : undefined
  const mainHeight = isExpanded
    ? Math.max(300, (modalHeightBudget ?? 540) - 128)
    : isCompact
      ? 220
      : 284
  const lensGap = isCompact ? 32 : 38
  const lensHeight = isExpanded ? 88 : isCompact ? 54 : 68
  const innerWidth = width - margin.left - margin.right
  const lensTop = margin.top + mainHeight + lensGap
  const height = lensTop + lensHeight + margin.bottom
  const model = buildModel(state)
  const points = makeCurvePoints(state, model)
  const probe = modelPrediction(state.probeX, state, model)
  const xScale = scaleLinear().domain(domain).range([0, innerWidth])
  const yScale = scaleLinear().domain(yExtent(points)).range([mainHeight, 0]).nice()
  const lensX = scaleLinear().domain(domain).range([0, innerWidth])
  const lensXs = grid(lensPointCount)
  const unitCurves = model.centers.map((center, index) => {
    const weight = model.networkWeights[index + 2] ?? 0
    return lensXs.map((x) => ({
      value: weight * hiddenFeature(x, center, state.activationKey),
      x,
    }))
  })
  const lensMax = Math.max(
    0.04,
    ...unitCurves.flatMap((unit) => unit.map((point) => Math.abs(point.value))),
  )
  const lensY = scaleLinear().domain([-lensMax, lensMax]).range([lensHeight, 0])
  const targetLine = d3Line<CurvePoint>()
    .x((point) => xScale(point.x))
    .y((point) => yScale(point.target))
    .curve(curveMonotoneX)
  const linearLine = d3Line<CurvePoint>()
    .x((point) => xScale(point.x))
    .y((point) => yScale(point.linear))
    .curve(curveMonotoneX)
  const networkLine = d3Line<CurvePoint>()
    .x((point) => xScale(point.x))
    .y((point) => yScale(point.network))
    .curve(curveMonotoneX)
  const errorArea = d3Area<CurvePoint>()
    .x((point) => xScale(point.x))
    .y0((point) => yScale(point.target))
    .y1((point) => yScale(point.network))
    .curve(curveMonotoneX)
  const unitLine = d3Line<UnitPoint>()
    .x((point) => lensX(point.x))
    .y((point) => lensY(point.value))
    .curve(curveMonotoneX)

  updateReadout(figure, state, points, probe)
  plot.replaceChildren()

  const svg = select(plot)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img")
    .attr(
      "aria-label",
      "Interactive universal approximation figure comparing a target function, a linear fit, and a shallow neural network approximation",
    )

  const main = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`)
  const lens = svg.append("g").attr("transform", `translate(${margin.left},${lensTop})`)

  main
    .append("rect")
    .attr("class", "universal-approximation-panel-bg")
    .attr("width", innerWidth)
    .attr("height", mainHeight)
    .attr("rx", 5)

  main
    .append("g")
    .attr("class", "universal-approximation-grid")
    .attr("transform", `translate(0,${mainHeight})`)
    .call(
      axisBottom(xScale)
        .ticks(isCompact ? 4 : 6)
        .tickSize(-mainHeight)
        .tickFormat(() => ""),
    )

  main
    .append("g")
    .attr("class", "universal-approximation-grid")
    .call(
      axisLeft(yScale)
        .ticks(isCompact ? 4 : 5)
        .tickSize(-innerWidth)
        .tickFormat(() => ""),
    )

  main
    .append("path")
    .datum(points)
    .attr("class", "universal-approximation-error-area")
    .attr("d", errorArea)

  main
    .append("path")
    .datum(points)
    .attr("class", "universal-approximation-target-line")
    .attr("d", targetLine)

  main
    .append("path")
    .datum(points)
    .attr("class", "universal-approximation-linear-line")
    .attr("d", linearLine)

  main
    .append("path")
    .datum(points)
    .attr("class", "universal-approximation-network-line")
    .attr("d", networkLine)

  main
    .append("g")
    .attr("class", "universal-approximation-axis")
    .attr("transform", `translate(0,${mainHeight})`)
    .call(
      axisBottom(xScale)
        .ticks(isCompact ? 4 : 6)
        .tickSizeOuter(0),
    )

  main
    .append("g")
    .attr("class", "universal-approximation-axis")
    .call(
      axisLeft(yScale)
        .ticks(isCompact ? 4 : 5)
        .tickSizeOuter(0),
    )

  main
    .append("text")
    .attr("class", "universal-approximation-panel-label")
    .attr("x", 0)
    .attr("y", -9)
    .text(targetDefinitions[state.targetKey].label)

  const probeX = xScale(probe.x)
  const probeTargetY = yScale(probe.target)
  const probeNetworkY = yScale(probe.network)

  main
    .append("line")
    .attr("class", "universal-approximation-probe-line")
    .attr("x1", probeX)
    .attr("x2", probeX)
    .attr("y1", 0)
    .attr("y2", mainHeight)

  main
    .append("line")
    .attr("class", "universal-approximation-probe-error")
    .attr("x1", probeX)
    .attr("x2", probeX)
    .attr("y1", probeTargetY)
    .attr("y2", probeNetworkY)

  main
    .append("circle")
    .attr("class", "universal-approximation-probe-target")
    .attr("cx", probeX)
    .attr("cy", probeTargetY)
    .attr("r", isCompact ? 3.5 : 4.2)

  main
    .append("circle")
    .attr("class", "universal-approximation-probe-network")
    .attr("cx", probeX)
    .attr("cy", probeNetworkY)
    .attr("r", isCompact ? 3.5 : 4.2)

  main
    .append("rect")
    .attr("class", "universal-approximation-probe-surface")
    .attr("width", innerWidth)
    .attr("height", mainHeight)

  lens
    .append("rect")
    .attr("class", "universal-approximation-lens-bg")
    .attr("width", innerWidth)
    .attr("height", lensHeight)
    .attr("rx", 5)

  lens
    .append("line")
    .attr("class", "universal-approximation-lens-zero")
    .attr("x1", 0)
    .attr("x2", innerWidth)
    .attr("y1", lensY(0))
    .attr("y2", lensY(0))

  lens
    .selectAll("path.universal-approximation-hidden-unit")
    .data(unitCurves)
    .join("path")
    .attr("class", "universal-approximation-hidden-unit")
    .attr("d", unitLine)
    .style("opacity", state.width > 20 ? 0.22 : 0.32)

  lens
    .append("g")
    .attr("class", "universal-approximation-lens-axis")
    .attr("transform", `translate(0,${lensHeight})`)
    .call(
      axisBottom(lensX)
        .ticks(isCompact ? 4 : 6)
        .tickSizeOuter(0),
    )

  lens
    .append("text")
    .attr("class", "universal-approximation-lens-label")
    .attr("x", 0)
    .attr("y", -8)
    .text("hidden unit contributions")
}

export const universalApproximationFigure: InteractiveFigureDefinition<UniversalApproximationState> =
  {
    bindControls: ({ addCleanup, figure, render, state }) => {
      const targetSelect = figure.querySelector<HTMLSelectElement>(
        ".universal-approximation-target",
      )
      const activationSelect = figure.querySelector<HTMLSelectElement>(
        ".universal-approximation-activation",
      )
      const widthSlider = figure.querySelector<HTMLInputElement>(".universal-approximation-width")
      const plot = figure.querySelector<HTMLElement>(".universal-approximation-plot")
      const updateTarget = () => {
        if (!targetSelect) return

        state.targetKey = readTargetKey(targetSelect.value)
        render()
      }
      const updateActivation = () => {
        if (!activationSelect) return

        state.activationKey = readActivationKey(activationSelect.value)
        render()
      }
      const updateWidth = () => {
        if (!widthSlider) return

        state.width = clampWidth(Number(widthSlider.value))
        render()
      }
      const updateProbe = (event: PointerEvent) => {
        const surface = plot?.querySelector<SVGRectElement>(
          ".universal-approximation-probe-surface",
        )
        if (!surface) return

        const bounds = surface.getBoundingClientRect()
        if (
          event.clientX < bounds.left ||
          event.clientX > bounds.right ||
          event.clientY < bounds.top ||
          event.clientY > bounds.bottom
        ) {
          return
        }

        const ratio = (event.clientX - bounds.left) / Math.max(bounds.width, 1)
        state.probeX = domain[0] + ratio * (domain[1] - domain[0])
        render()
      }

      targetSelect?.addEventListener("change", updateTarget)
      activationSelect?.addEventListener("change", updateActivation)
      widthSlider?.addEventListener("input", updateWidth)
      plot?.addEventListener("pointermove", updateProbe)
      addCleanup(() => {
        targetSelect?.removeEventListener("change", updateTarget)
        activationSelect?.removeEventListener("change", updateActivation)
        widthSlider?.removeEventListener("input", updateWidth)
        plot?.removeEventListener("pointermove", updateProbe)
      })
    },
    classNames: ["universal-approximation-figure"],
    cloneState: cloneUniversalApproximationState,
    expandIgnoreSelector: "button, input, label, select, a, [data-no-expand]",
    readState: readUniversalApproximationState,
    render: renderUniversalApproximation,
    resample: resetUniversalApproximation,
    serializeState: (state) => ({
      activation: state.activationKey,
      probe: state.probeX,
      target: state.targetKey,
      width: state.width,
    }),
    syncControls: syncUniversalApproximationControls,
    template: universalApproximationTemplate,
    type: "universal-approximation",
  }

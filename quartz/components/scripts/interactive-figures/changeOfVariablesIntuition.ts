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
import { normalPdf, readNumber } from "./math"
import { expandButtonHtml } from "./ui"

type TransformKey = "shift" | "compress" | "stretch" | "exp" | "cubic"

type TransformDefinition = {
  cue: string
  derivative: (omega: number) => number
  key: TransformKey
  label: string
  transform: (omega: number) => number
}

type ChangeOfVariablesState = {
  intervalWidth: number
  omega: number
  transformKey: TransformKey
}

const sourceDomain: [number, number] = [-3, 3]
const omegaDomain: [number, number] = [-2.5, 2.5]
const intervalWidthDomain: [number, number] = [0.16, 0.8]
const sliderMax = 1000
const curvePointCount = 220

const transformDefinitions: Record<TransformKey, TransformDefinition> = {
  shift: {
    cue: "move only",
    derivative: () => 1,
    key: "shift",
    label: "x=ω+1",
    transform: (omega) => omega + 1,
  },
  compress: {
    cue: "compress",
    derivative: () => 0.5,
    key: "compress",
    label: "x=0.5ω",
    transform: (omega) => 0.5 * omega,
  },
  stretch: {
    cue: "stretch",
    derivative: () => 2,
    key: "stretch",
    label: "x=2ω",
    transform: (omega) => 2 * omega,
  },
  exp: {
    cue: "nonlinear stretch",
    derivative: Math.exp,
    key: "exp",
    label: "x=exp(ω)",
    transform: Math.exp,
  },
  cubic: {
    cue: "nonlinear stretch",
    derivative: (omega) => 1 + 0.48 * omega * omega,
    key: "cubic",
    label: "x=ω+0.16ω³",
    transform: (omega) => omega + 0.16 * omega * omega * omega,
  },
}

const transformKeys: TransformKey[] = ["shift", "compress", "stretch", "exp", "cubic"]

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function readTransformKey(value: string | undefined): TransformKey {
  return transformKeys.includes(value as TransformKey) ? (value as TransformKey) : "exp"
}

function sliderValue(value: number, domain: [number, number]) {
  const ratio = (clamp(value, domain[0], domain[1]) - domain[0]) / (domain[1] - domain[0])
  return Math.round(ratio * sliderMax)
}

function valueFromSlider(value: number, domain: [number, number]) {
  const ratio = clamp(value / sliderMax, 0, 1)
  return domain[0] + ratio * (domain[1] - domain[0])
}

function formatNumber(value: number, digits = 3) {
  if (!Number.isFinite(value)) return "inf"

  return Number.parseFloat(value.toFixed(digits)).toString()
}

function cssColor(element: HTMLElement, name: string, fallback: string) {
  const color = getComputedStyle(element).getPropertyValue(name).trim()
  return color || fallback
}

function intervalGrid(start: number, end: number, count: number) {
  return range(0, count).map((index) => start + ((end - start) * index) / (count - 1))
}

function targetDomainForTransform(transform: TransformDefinition): [number, number] {
  const left = transform.transform(sourceDomain[0])
  const right = transform.transform(sourceDomain[1])

  return transform.key === "exp" ? [0, right] : [left, right]
}

function readChangeOfVariablesState(figure: HTMLElement): ChangeOfVariablesState {
  return {
    intervalWidth: clamp(
      readNumber(figure.dataset.width, 0.36),
      intervalWidthDomain[0],
      intervalWidthDomain[1],
    ),
    omega: clamp(readNumber(figure.dataset.omega, 0.6), omegaDomain[0], omegaDomain[1]),
    transformKey: readTransformKey(figure.dataset.transform),
  }
}

function cloneChangeOfVariablesState(state: ChangeOfVariablesState): ChangeOfVariablesState {
  return { ...state }
}

function syncChangeOfVariablesControls(figure: HTMLElement, state: ChangeOfVariablesState) {
  const transformSelect = figure.querySelector<HTMLSelectElement>(".change-variables-transform")
  const omegaSlider = figure.querySelector<HTMLInputElement>(".change-variables-omega-slider")
  const widthSlider = figure.querySelector<HTMLInputElement>(".change-variables-width-slider")
  const omegaOutput = figure.querySelector<HTMLOutputElement>(".change-variables-omega")
  const widthOutput = figure.querySelector<HTMLOutputElement>(".change-variables-width")
  const xOutput = figure.querySelector<HTMLElement>(".change-variables-x0")
  const dxOutput = figure.querySelector<HTMLElement>(".change-variables-dx")
  const densityOutput = figure.querySelector<HTMLElement>(".change-variables-density")
  const derivativeOutput = figure.querySelector<HTMLElement>(".change-variables-derivative")
  const transform = transformDefinitions[state.transformKey]
  const derivative = Math.abs(transform.derivative(state.omega))
  const sourceDensity = normalPdf(state.omega, 0, 1)
  const targetDensity = sourceDensity / derivative
  const mappedWidth = derivative * state.intervalWidth

  if (transformSelect) transformSelect.value = state.transformKey
  if (omegaSlider) omegaSlider.value = String(sliderValue(state.omega, omegaDomain))
  if (widthSlider) widthSlider.value = String(sliderValue(state.intervalWidth, intervalWidthDomain))
  if (omegaOutput) omegaOutput.value = formatNumber(state.omega, 2)
  if (widthOutput) widthOutput.value = formatNumber(state.intervalWidth, 2)
  if (xOutput) xOutput.textContent = `x0 ${formatNumber(transform.transform(state.omega))}`
  if (dxOutput) dxOutput.textContent = `dx ${formatNumber(mappedWidth)}`
  if (densityOutput) {
    densityOutput.textContent = `pX ${formatNumber(targetDensity)}`
  }
  if (derivativeOutput) {
    derivativeOutput.textContent = `scale ${formatNumber(derivative)}`
  }

  figure.dataset.transform = state.transformKey
  figure.dataset.omega = String(state.omega)
  figure.dataset.width = String(state.intervalWidth)
}

function changeOfVariablesTemplate({
  canExpand,
  id,
  state,
}: {
  canExpand: boolean
  id: string
  state: ChangeOfVariablesState
}) {
  const options = transformKeys
    .map((key) => {
      const selected = key === state.transformKey ? " selected" : ""
      return `<option value="${key}"${selected}>${transformDefinitions[key].label}</option>`
    })
    .join("")

  return `
    <div class="interactive-figure-toolbar change-variables-toolbar">
      <div class="interactive-figure-title">Change of variables intuition</div>
      <div class="interactive-figure-actions">
        ${canExpand ? expandButtonHtml() : ""}
      </div>
    </div>
    <div class="interactive-figure-plot change-variables-plot"></div>
    <div class="change-variables-readout" aria-live="polite">
      <span class="change-variables-x0"></span>
      <span class="change-variables-derivative"></span>
      <span class="change-variables-dx"></span>
      <span class="change-variables-density"></span>
    </div>
    <div class="change-variables-controls" data-no-expand>
      <label>
        <span>map</span>
        <select class="change-variables-transform" aria-label="Transformation">
          ${options}
        </select>
      </label>
      <label>
        <span>ω₀</span>
        <input class="change-variables-omega-slider" type="range" min="0" max="${sliderMax}" step="1" value="${sliderValue(
          state.omega,
          omegaDomain,
        )}" aria-describedby="${id}-omega" />
        <output class="change-variables-omega" id="${id}-omega">${formatNumber(
          state.omega,
          2,
        )}</output>
      </label>
      <label>
        <span>dω</span>
        <input class="change-variables-width-slider" type="range" min="0" max="${sliderMax}" step="1" value="${sliderValue(
          state.intervalWidth,
          intervalWidthDomain,
        )}" aria-describedby="${id}-width" />
        <output class="change-variables-width" id="${id}-width">${formatNumber(
          state.intervalWidth,
          2,
        )}</output>
      </label>
    </div>
  `
}

function renderChangeOfVariables(figure: HTMLElement, state: ChangeOfVariablesState) {
  const plot = figure.querySelector<HTMLElement>(".change-variables-plot")
  if (!plot) return

  const width = Math.max(plot.clientWidth, 300)
  const isExpanded = figure.classList.contains("is-expanded")
  const isCompact = width < 560 && !isExpanded
  const plotHeight = isExpanded ? 168 : isCompact ? 116 : 132
  const verticalGap = isExpanded ? 108 : isCompact ? 84 : 94
  const sourceTop = isExpanded ? 26 : 22
  const targetTop = sourceTop + plotHeight + verticalGap
  const height = targetTop + plotHeight + (isExpanded ? 54 : 44)
  const margin = {
    left: isCompact ? 42 : 52,
    right: isCompact ? 16 : 30,
  }
  const innerWidth = width - margin.left - margin.right
  const transform = transformDefinitions[state.transformKey]
  const sourceGrid = intervalGrid(sourceDomain[0], sourceDomain[1], curvePointCount)
  const targetDomain = targetDomainForTransform(transform)
  const sourceCurve = sourceGrid.map((omega) => [omega, normalPdf(omega, 0, 1)] as [number, number])
  const targetCurve = sourceGrid.map((omega) => {
    const derivative = Math.abs(transform.derivative(omega))
    return [transform.transform(omega), normalPdf(omega, 0, 1) / derivative] as [number, number]
  })
  const targetMax = Math.max(...targetCurve.map((point) => point[1]))
  const halfWidth = state.intervalWidth / 2
  const omegaLeft = state.omega - halfWidth
  const omegaRight = state.omega + halfWidth
  const xLeft = transform.transform(omegaLeft)
  const xRight = transform.transform(omegaRight)
  const x0 = transform.transform(state.omega)
  const derivative = Math.abs(transform.derivative(state.omega))
  const sourceDensity = normalPdf(state.omega, 0, 1)
  const targetDensity = sourceDensity / derivative
  const intervalOmegas = intervalGrid(omegaLeft, omegaRight, 32)
  const sourceInterval = intervalOmegas.map(
    (omega) => [omega, normalPdf(omega, 0, 1)] as [number, number],
  )
  const targetInterval = intervalOmegas.map((omega) => {
    const localDerivative = Math.abs(transform.derivative(omega))
    return [transform.transform(omega), normalPdf(omega, 0, 1) / localDerivative] as [
      number,
      number,
    ]
  })

  const sourceX = scaleLinear().domain(sourceDomain).range([0, innerWidth])
  const sourceY = scaleLinear().domain([0, 0.44]).range([plotHeight, 0])
  const targetX = scaleLinear().domain(targetDomain).range([0, innerWidth]).nice()
  const targetY = scaleLinear()
    .domain([0, targetMax * 1.16])
    .range([plotHeight, 0])
    .nice()
  const sourceLine = d3Line<[number, number]>()
    .x((point) => sourceX(point[0]))
    .y((point) => sourceY(point[1]))
    .curve(curveMonotoneX)
  const targetLine = d3Line<[number, number]>()
    .x((point) => targetX(point[0]))
    .y((point) => targetY(point[1]))
    .curve(curveMonotoneX)
  const sourceArea = d3Area<[number, number]>()
    .x((point) => sourceX(point[0]))
    .y0(plotHeight)
    .y1((point) => sourceY(point[1]))
    .curve(curveMonotoneX)
  const targetArea = d3Area<[number, number]>()
    .x((point) => targetX(point[0]))
    .y0(plotHeight)
    .y1((point) => targetY(point[1]))
    .curve(curveMonotoneX)
  const sourceColor = cssColor(figure, "--cov-source", "#2c5f9e")
  const targetColor = cssColor(figure, "--cov-target", "#9f3d46")
  const massColor = cssColor(figure, "--cov-mass", "#d09a27")
  const ribbonColor = cssColor(figure, "--cov-ribbon", "rgba(208, 154, 39, 0.2)")
  const axisColor = cssColor(figure, "--chart-axis", "#726a60")
  const gridColor = cssColor(figure, "--chart-grid", "#e7e0d6")
  const sourceIntervalLeftX = margin.left + sourceX(omegaLeft)
  const sourceIntervalRightX = margin.left + sourceX(omegaRight)
  const targetIntervalLeftX = margin.left + targetX(xLeft)
  const targetIntervalRightX = margin.left + targetX(xRight)
  const sourceBaseY = sourceTop + plotHeight
  const targetStartY = targetTop
  const ribbonInset = isCompact ? 8 : 12
  const ribbonPath = [
    `M${sourceIntervalLeftX},${sourceBaseY + ribbonInset}`,
    `C${sourceIntervalLeftX},${sourceBaseY + verticalGap * 0.42} ${targetIntervalLeftX},${
      targetStartY - verticalGap * 0.42
    } ${targetIntervalLeftX},${targetStartY - ribbonInset}`,
    `L${targetIntervalRightX},${targetStartY - ribbonInset}`,
    `C${targetIntervalRightX},${targetStartY - verticalGap * 0.42} ${sourceIntervalRightX},${
      sourceBaseY + verticalGap * 0.42
    } ${sourceIntervalRightX},${sourceBaseY + ribbonInset}`,
    "Z",
  ].join(" ")
  const densityLabel =
    Math.abs(derivative - 1) < 0.05 ? "same height" : derivative > 1 ? "density down" : "density up"

  plot.replaceChildren()

  const svg = select(plot)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img")
    .attr(
      "aria-label",
      "Interactive change of variables figure showing conserved probability mass under a transformation",
    )

  const sourceGroup = svg
    .append("g")
    .attr("class", "change-variables-panel")
    .attr("transform", `translate(${margin.left},${sourceTop})`)
  const targetGroup = svg
    .append("g")
    .attr("class", "change-variables-panel")
    .attr("transform", `translate(${margin.left},${targetTop})`)

  sourceGroup
    .append("g")
    .attr("class", "change-variables-grid")
    .call(
      axisLeft(sourceY)
        .ticks(3)
        .tickSize(-innerWidth)
        .tickFormat(() => ""),
    )
    .call((grid) => grid.select(".domain").remove())
    .call((grid) => grid.selectAll("line").attr("stroke", gridColor))

  targetGroup
    .append("g")
    .attr("class", "change-variables-grid")
    .call(
      axisLeft(targetY)
        .ticks(3)
        .tickSize(-innerWidth)
        .tickFormat(() => ""),
    )
    .call((grid) => grid.select(".domain").remove())
    .call((grid) => grid.selectAll("line").attr("stroke", gridColor))

  sourceGroup
    .append("path")
    .datum(sourceInterval)
    .attr("class", "change-variables-mass-area")
    .attr("d", sourceArea)
    .attr("fill", massColor)

  targetGroup
    .append("path")
    .datum(targetInterval)
    .attr("class", "change-variables-mass-area")
    .attr("d", targetArea)
    .attr("fill", massColor)

  sourceGroup
    .append("path")
    .datum(sourceCurve)
    .attr("class", "change-variables-density-line")
    .attr("d", sourceLine)
    .attr("fill", "none")
    .attr("stroke", sourceColor)

  targetGroup
    .append("path")
    .datum(targetCurve)
    .attr("class", "change-variables-density-line")
    .attr("d", targetLine)
    .attr("fill", "none")
    .attr("stroke", targetColor)

  svg
    .append("path")
    .attr("class", "change-variables-ribbon")
    .attr("d", ribbonPath)
    .attr("fill", ribbonColor)

  sourceGroup
    .append("line")
    .attr("class", "change-variables-marker-line")
    .attr("x1", sourceX(state.omega))
    .attr("x2", sourceX(state.omega))
    .attr("y1", sourceY(sourceDensity))
    .attr("y2", plotHeight)

  targetGroup
    .append("line")
    .attr("class", "change-variables-marker-line")
    .attr("x1", targetX(x0))
    .attr("x2", targetX(x0))
    .attr("y1", targetY(targetDensity))
    .attr("y2", plotHeight)

  sourceGroup
    .append("circle")
    .attr("class", "change-variables-source-point")
    .attr("cx", sourceX(state.omega))
    .attr("cy", sourceY(sourceDensity))
    .attr("r", isExpanded ? 4.8 : 4)
    .attr("fill", sourceColor)

  targetGroup
    .append("circle")
    .attr("class", "change-variables-target-point")
    .attr("cx", targetX(x0))
    .attr("cy", targetY(targetDensity))
    .attr("r", isExpanded ? 4.8 : 4)
    .attr("fill", targetColor)

  sourceGroup
    .append("g")
    .attr("class", "change-variables-axis")
    .attr("transform", `translate(0,${plotHeight})`)
    .call(
      axisBottom(sourceX)
        .ticks(isCompact ? 5 : 7)
        .tickSizeOuter(0),
    )

  sourceGroup
    .append("g")
    .attr("class", "change-variables-axis")
    .call(
      axisLeft(sourceY)
        .ticks(isCompact ? 3 : 4)
        .tickSizeOuter(0),
    )

  targetGroup
    .append("g")
    .attr("class", "change-variables-axis")
    .attr("transform", `translate(0,${plotHeight})`)
    .call(
      axisBottom(targetX)
        .ticks(isCompact ? 4 : 6)
        .tickSizeOuter(0),
    )

  targetGroup
    .append("g")
    .attr("class", "change-variables-axis")
    .call(
      axisLeft(targetY)
        .ticks(isCompact ? 3 : 4)
        .tickSizeOuter(0),
    )

  sourceGroup
    .append("text")
    .attr("class", "change-variables-panel-label")
    .attr("x", 0)
    .attr("y", -8)
    .text("source")

  targetGroup
    .append("text")
    .attr("class", "change-variables-panel-label")
    .attr("x", 0)
    .attr("y", -8)
    .text("target")

  svg
    .append("text")
    .attr("class", "change-variables-ribbon-label")
    .attr("x", width / 2)
    .attr("y", sourceBaseY + verticalGap * 0.42)
    .text(isCompact ? "same mass" : "mass conserved")

  svg
    .append("text")
    .attr("class", "change-variables-stretch-label")
    .attr("x", width / 2)
    .attr("y", sourceBaseY + verticalGap * 0.42 + 19)
    .text(`${transform.cue}: ${densityLabel}`)

  svg
    .selectAll(".change-variables-axis path, .change-variables-axis line")
    .attr("stroke", axisColor)
  svg.selectAll(".change-variables-axis text").attr("fill", axisColor)
}

export const changeOfVariablesIntuitionFigure: InteractiveFigureDefinition<ChangeOfVariablesState> =
  {
    bindControls: ({ addCleanup, figure, render, state }) => {
      const transformSelect = figure.querySelector<HTMLSelectElement>(".change-variables-transform")
      const omegaSlider = figure.querySelector<HTMLInputElement>(".change-variables-omega-slider")
      const widthSlider = figure.querySelector<HTMLInputElement>(".change-variables-width-slider")
      const updateTransform = () => {
        if (!transformSelect) return

        state.transformKey = readTransformKey(transformSelect.value)
        render()
      }
      const updateOmega = () => {
        if (!omegaSlider) return

        state.omega = valueFromSlider(Number(omegaSlider.value), omegaDomain)
        render()
      }
      const updateWidth = () => {
        if (!widthSlider) return

        state.intervalWidth = valueFromSlider(Number(widthSlider.value), intervalWidthDomain)
        render()
      }

      transformSelect?.addEventListener("change", updateTransform)
      omegaSlider?.addEventListener("input", updateOmega)
      widthSlider?.addEventListener("input", updateWidth)
      addCleanup(() => {
        transformSelect?.removeEventListener("change", updateTransform)
        omegaSlider?.removeEventListener("input", updateOmega)
        widthSlider?.removeEventListener("input", updateWidth)
      })
    },
    classNames: ["change-variables-figure"],
    cloneState: cloneChangeOfVariablesState,
    expandIgnoreSelector: "button, input, label, select, a, [data-no-expand]",
    readState: readChangeOfVariablesState,
    render: renderChangeOfVariables,
    serializeState: (state) => ({
      omega: state.omega,
      transform: state.transformKey,
      width: state.intervalWidth,
    }),
    syncControls: syncChangeOfVariablesControls,
    template: changeOfVariablesTemplate,
    type: "change-of-variables-intuition",
  }

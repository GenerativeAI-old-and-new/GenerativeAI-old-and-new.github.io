import {
  area as d3Area,
  axisBottom,
  axisLeft,
  curveMonotoneX,
  line as d3Line,
  max,
  range,
  scaleLinear,
  select,
} from "d3"
import { InteractiveFigureDefinition } from "./core"
import { readNumber } from "./math"
import { expandButtonHtml } from "./ui"

type DistributionKey = "logistic" | "exponential" | "power"

type DistributionDefinition = {
  cdf: (x: number) => number
  domain: [number, number]
  inverse: (u: number) => number
  key: DistributionKey
  label: string
  pdf: (x: number) => number
}

type InverseCdfState = {
  distributionKey: DistributionKey
  u: number
}

const sliderMax = 1000
const uDomain: [number, number] = [0.02, 0.98]
const curvePointCount = 220
let markerCount = 0

const distributionDefinitions: Record<DistributionKey, DistributionDefinition> = {
  logistic: {
    cdf: (x) => 1 / (1 + Math.exp(-x)),
    domain: [-6, 6],
    inverse: (u) => Math.log(u / (1 - u)),
    key: "logistic",
    label: "Logistic",
    pdf: (x) => {
      const f = 1 / (1 + Math.exp(-x))
      return f * (1 - f)
    },
  },
  exponential: {
    cdf: (x) => 1 - Math.exp(-x),
    domain: [0, 5],
    inverse: (u) => -Math.log(1 - u),
    key: "exponential",
    label: "Exponential",
    pdf: (x) => Math.exp(-x),
  },
  power: {
    cdf: (x) => x ** 2.4,
    domain: [0, 1],
    inverse: (u) => u ** (1 / 2.4),
    key: "power",
    label: "Skewed [0,1]",
    pdf: (x) => 2.4 * x ** 1.4,
  },
}

const distributionKeys: DistributionKey[] = ["logistic", "exponential", "power"]

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function readDistributionKey(value: string | undefined): DistributionKey {
  return distributionKeys.includes(value as DistributionKey)
    ? (value as DistributionKey)
    : "logistic"
}

function sliderValue(value: number) {
  const ratio = (clamp(value, uDomain[0], uDomain[1]) - uDomain[0]) / (uDomain[1] - uDomain[0])
  return Math.round(ratio * sliderMax)
}

function valueFromSlider(value: number) {
  const ratio = clamp(value / sliderMax, 0, 1)
  return uDomain[0] + ratio * (uDomain[1] - uDomain[0])
}

function formatNumber(value: number, digits = 2) {
  return Number.parseFloat(value.toFixed(digits)).toString()
}

function domainGrid([left, right]: [number, number], count: number) {
  return range(0, count).map((index) => left + ((right - left) * index) / (count - 1))
}

function readInverseCdfState(figure: HTMLElement): InverseCdfState {
  return {
    distributionKey: readDistributionKey(figure.dataset.distribution),
    u: clamp(readNumber(figure.dataset.u, 0.72), uDomain[0], uDomain[1]),
  }
}

function cloneInverseCdfState(state: InverseCdfState): InverseCdfState {
  return { ...state }
}

function syncInverseCdfControls(figure: HTMLElement, state: InverseCdfState) {
  const distributionSelect = figure.querySelector<HTMLSelectElement>(".inverse-cdf-distribution")
  const uSlider = figure.querySelector<HTMLInputElement>(".inverse-cdf-u-slider")
  const uOutput = figure.querySelector<HTMLOutputElement>(".inverse-cdf-u")
  const uReadout = figure.querySelector<HTMLElement>(".inverse-cdf-u-readout")
  const xOutput = figure.querySelector<HTMLElement>(".inverse-cdf-x")
  const areaOutput = figure.querySelector<HTMLElement>(".inverse-cdf-area")
  const distribution = distributionDefinitions[state.distributionKey]
  const x = distribution.inverse(state.u)
  const digits = distribution.domain[1] - distribution.domain[0] <= 1 ? 3 : 2

  if (distributionSelect) distributionSelect.value = state.distributionKey
  if (uSlider) uSlider.value = String(sliderValue(state.u))
  if (uOutput) uOutput.value = formatNumber(state.u, 2)
  if (uReadout) uReadout.textContent = `U ${formatNumber(state.u, 2)}`
  if (xOutput) xOutput.textContent = `x ${formatNumber(x, digits)}`
  if (areaOutput) areaOutput.textContent = `area ${formatNumber(state.u, 2)}`

  figure.dataset.distribution = state.distributionKey
  figure.dataset.u = String(state.u)
}

function inverseCdfTemplate({
  canExpand,
  id,
  state,
}: {
  canExpand: boolean
  id: string
  state: InverseCdfState
}) {
  const options = distributionKeys
    .map((key) => {
      const selected = key === state.distributionKey ? " selected" : ""
      return `<option value="${key}"${selected}>${distributionDefinitions[key].label}</option>`
    })
    .join("")

  return `
    <div class="interactive-figure-toolbar inverse-cdf-toolbar">
      <div class="interactive-figure-title">Inverse CDF transform</div>
      <div class="interactive-figure-actions">
        ${canExpand ? expandButtonHtml() : ""}
      </div>
    </div>
    <div class="interactive-figure-plot inverse-cdf-plot"></div>
    <div class="inverse-cdf-readout" aria-live="polite">
      <span class="inverse-cdf-u-readout"></span>
      <span class="inverse-cdf-x"></span>
      <span class="inverse-cdf-area"></span>
    </div>
    <div class="inverse-cdf-controls" data-no-expand>
      <label>
        <span>dist</span>
        <select class="inverse-cdf-distribution" aria-label="Distribution">
          ${options}
        </select>
      </label>
      <label>
        <span>U</span>
        <input class="inverse-cdf-u-slider" type="range" min="0" max="${sliderMax}" step="1" value="${sliderValue(
          state.u,
        )}" aria-describedby="${id}-u" />
        <output class="inverse-cdf-u" id="${id}-u">${formatNumber(state.u, 2)}</output>
      </label>
    </div>
  `
}

function renderInverseCdf(figure: HTMLElement, state: InverseCdfState) {
  const plot = figure.querySelector<HTMLElement>(".inverse-cdf-plot")
  if (!plot) return

  const width = Math.max(plot.clientWidth, 300)
  const isExpanded = figure.classList.contains("is-expanded")
  const isCompact = width < 560 && !isExpanded
  const margin = {
    bottom: isExpanded ? 38 : 32,
    left: isCompact ? 42 : 54,
    right: isCompact ? 14 : 28,
    top: isExpanded ? 22 : 18,
  }
  const cdfHeight = isExpanded ? 245 : isCompact ? 178 : 210
  const densityHeight = isExpanded ? 108 : isCompact ? 76 : 90
  const panelGap = isExpanded ? 56 : isCompact ? 42 : 48
  const cdfTop = margin.top
  const densityTop = cdfTop + cdfHeight + panelGap
  const height = densityTop + densityHeight + margin.bottom
  const innerWidth = width - margin.left - margin.right
  const distribution = distributionDefinitions[state.distributionKey]
  const domain = distribution.domain
  const x = distribution.inverse(state.u)
  const markerId = `inverse-cdf-arrowhead-${++markerCount}`
  const grid = domainGrid(domain, curvePointCount)
  const cdfPoints = grid.map((point) => [point, distribution.cdf(point)] as [number, number])
  const densityPoints = grid.map((point) => [point, distribution.pdf(point)] as [number, number])
  const selectedDensityPoints = densityPoints.filter((point) => point[0] <= x)

  if (
    selectedDensityPoints.length === 0 ||
    selectedDensityPoints[selectedDensityPoints.length - 1][0] < x
  ) {
    selectedDensityPoints.push([x, distribution.pdf(x)])
  }

  const xScale = scaleLinear().domain(domain).range([0, innerWidth])
  const cdfY = scaleLinear().domain([0, 1]).range([cdfHeight, 0])
  const densityY = scaleLinear()
    .domain([0, (max(densityPoints, (point) => point[1]) ?? 1) * 1.15])
    .range([densityHeight, 0])
    .nice()
  const cdfLine = d3Line<[number, number]>()
    .x((point) => xScale(point[0]))
    .y((point) => cdfY(point[1]))
    .curve(curveMonotoneX)
  const densityLine = d3Line<[number, number]>()
    .x((point) => xScale(point[0]))
    .y((point) => densityY(point[1]))
    .curve(curveMonotoneX)
  const densityArea = d3Area<[number, number]>()
    .x((point) => xScale(point[0]))
    .y0(densityHeight)
    .y1((point) => densityY(point[1]))
    .curve(curveMonotoneX)
  const xPosition = xScale(x)
  const uPosition = cdfY(state.u)

  plot.replaceChildren()

  const svg = select(plot)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img")
    .attr(
      "aria-label",
      "Inverse CDF transform diagram showing a uniform probability level mapped through the inverse CDF to a sample",
    )

  svg
    .append("defs")
    .append("marker")
    .attr("id", markerId)
    .attr("viewBox", "0 0 10 10")
    .attr("refX", 8)
    .attr("refY", 5)
    .attr("markerWidth", 7)
    .attr("markerHeight", 7)
    .attr("orient", "auto-start-reverse")
    .append("path")
    .attr("class", "inverse-cdf-arrow-marker")
    .attr("d", "M0,0 L10,5 L0,10 Z")

  const cdfGroup = svg
    .append("g")
    .attr("class", "inverse-cdf-cdf-panel")
    .attr("transform", `translate(${margin.left},${cdfTop})`)
  const densityGroup = svg
    .append("g")
    .attr("class", "inverse-cdf-density-panel")
    .attr("transform", `translate(${margin.left},${densityTop})`)

  cdfGroup
    .append("g")
    .attr("class", "inverse-cdf-grid")
    .call(
      axisLeft(cdfY)
        .ticks(4)
        .tickSize(-innerWidth)
        .tickFormat(() => ""),
    )

  densityGroup
    .append("g")
    .attr("class", "inverse-cdf-grid")
    .call(
      axisLeft(densityY)
        .ticks(3)
        .tickSize(-innerWidth)
        .tickFormat(() => ""),
    )

  densityGroup
    .append("path")
    .datum(densityPoints)
    .attr("class", "inverse-cdf-density-area is-full")
    .attr("d", densityArea)

  densityGroup
    .append("path")
    .datum(selectedDensityPoints)
    .attr("class", "inverse-cdf-density-area is-selected")
    .attr("d", densityArea)

  cdfGroup.append("path").datum(cdfPoints).attr("class", "inverse-cdf-cdf-line").attr("d", cdfLine)

  densityGroup
    .append("path")
    .datum(densityPoints)
    .attr("class", "inverse-cdf-density-line")
    .attr("d", densityLine)

  cdfGroup
    .append("path")
    .attr("class", "inverse-cdf-guide is-arrow")
    .attr("d", `M0,${uPosition} L${xPosition},${uPosition}`)
    .attr("marker-end", `url(#${markerId})`)

  cdfGroup
    .append("line")
    .attr("class", "inverse-cdf-guide")
    .attr("x1", xPosition)
    .attr("x2", xPosition)
    .attr("y1", uPosition)
    .attr("y2", cdfHeight)

  densityGroup
    .append("line")
    .attr("class", "inverse-cdf-guide")
    .attr("x1", xPosition)
    .attr("x2", xPosition)
    .attr("y1", 0)
    .attr("y2", densityHeight)

  cdfGroup
    .append("circle")
    .attr("class", "inverse-cdf-u-point")
    .attr("cx", 0)
    .attr("cy", uPosition)
    .attr("r", isExpanded ? 4.8 : 4.2)

  cdfGroup
    .append("circle")
    .attr("class", "inverse-cdf-curve-point")
    .attr("cx", xPosition)
    .attr("cy", uPosition)
    .attr("r", isExpanded ? 4.8 : 4.2)

  cdfGroup
    .append("text")
    .attr("class", "inverse-cdf-panel-label")
    .attr("x", 0)
    .attr("y", -7)
    .text("CDF F(x)")

  densityGroup
    .append("text")
    .attr("class", "inverse-cdf-panel-label")
    .attr("x", 0)
    .attr("y", -7)
    .text("density p(x)")

  cdfGroup
    .append("text")
    .attr("class", "inverse-cdf-step-label")
    .attr("x", Math.min(innerWidth - 84, Math.max(8, xPosition * 0.5 - 38)))
    .attr("y", Math.max(14, uPosition - 10))
    .text("draw U")

  cdfGroup
    .append("text")
    .attr("class", "inverse-cdf-step-label")
    .attr("x", Math.min(innerWidth - 112, Math.max(8, xPosition + 8)))
    .attr("y", cdfHeight - 12)
    .text("return x")

  densityGroup
    .append("text")
    .attr("class", "inverse-cdf-step-label")
    .attr("x", Math.min(innerWidth - 110, Math.max(8, xPosition * 0.5 - 38)))
    .attr("y", 16)
    .text("area = U")

  cdfGroup
    .append("g")
    .attr("class", "inverse-cdf-axis")
    .attr("transform", `translate(0,${cdfHeight})`)
    .call(
      axisBottom(xScale)
        .ticks(isCompact ? 4 : 7)
        .tickSizeOuter(0),
    )

  cdfGroup
    .append("g")
    .attr("class", "inverse-cdf-axis")
    .call(
      axisLeft(cdfY)
        .ticks(isCompact ? 3 : 5)
        .tickSizeOuter(0),
    )

  densityGroup
    .append("g")
    .attr("class", "inverse-cdf-axis")
    .attr("transform", `translate(0,${densityHeight})`)
    .call(
      axisBottom(xScale)
        .ticks(isCompact ? 4 : 7)
        .tickSizeOuter(0),
    )

  densityGroup
    .append("g")
    .attr("class", "inverse-cdf-axis")
    .call(
      axisLeft(densityY)
        .ticks(isCompact ? 2 : 3)
        .tickSizeOuter(0),
    )
}

export const inverseCdfTransformFigure: InteractiveFigureDefinition<InverseCdfState> = {
  bindControls: ({ addCleanup, figure, render, state }) => {
    const distributionSelect = figure.querySelector<HTMLSelectElement>(".inverse-cdf-distribution")
    const uSlider = figure.querySelector<HTMLInputElement>(".inverse-cdf-u-slider")
    const updateDistribution = () => {
      if (!distributionSelect) return

      state.distributionKey = readDistributionKey(distributionSelect.value)
      render()
    }
    const updateU = () => {
      if (!uSlider) return

      state.u = valueFromSlider(Number(uSlider.value))
      render()
    }

    distributionSelect?.addEventListener("change", updateDistribution)
    uSlider?.addEventListener("input", updateU)
    addCleanup(() => {
      distributionSelect?.removeEventListener("change", updateDistribution)
      uSlider?.removeEventListener("input", updateU)
    })
  },
  classNames: ["inverse-cdf-figure"],
  cloneState: cloneInverseCdfState,
  readState: readInverseCdfState,
  render: renderInverseCdf,
  serializeState: (state) => ({
    distribution: state.distributionKey,
    u: state.u,
  }),
  syncControls: syncInverseCdfControls,
  template: inverseCdfTemplate,
  type: "inverse-cdf-transform",
}

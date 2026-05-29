import {
  area as d3Area,
  axisBottom,
  bin as d3Bin,
  curveBasis,
  line as d3Line,
  max,
  range,
  scaleLinear,
  select,
  type Selection,
} from "d3"
import { InteractiveFigureDefinition } from "./core"
import {
  clampSampleSize,
  mean,
  normalPdf,
  readNumber,
  sampleSizeToSliderValue,
  sampleSliderMax,
  seededRandom,
  sliderValueToSampleSize,
  standardDeviation,
} from "./math"
import { sampleTicksHtml, toolbarHtml } from "./ui"

type DistributionLearningState = {
  n: number
  samples: number[]
  seed: number
}

type Panel = {
  height: number
  left: number
  top: number
  width: number
}

const domain: [number, number] = [-4, 4]
const mixture = [
  { mu: -1.25, sigma: 0.44, weight: 0.58 },
  { mu: 1.35, sigma: 0.66, weight: 0.42 },
]
let markerCount = 0

function trueDensity(x: number) {
  return mixture.reduce(
    (total, component) => total + component.weight * normalPdf(x, component.mu, component.sigma),
    0,
  )
}

function gaussianFromRandom(random: () => number) {
  const u = Math.max(random(), Number.EPSILON)
  const v = random()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function makeMixtureSamples(n: number, seed: number) {
  const random = seededRandom(seed)
  const samples: number[] = []

  for (let i = 0; i < n; i++) {
    const component = random() < mixture[0].weight ? mixture[0] : mixture[1]
    samples.push(component.mu + component.sigma * gaussianFromRandom(random))
  }

  return samples
}

function readDistributionLearningState(figure: HTMLElement): DistributionLearningState {
  const n = clampSampleSize(readNumber(figure.dataset.n, 120))
  const seed = Math.round(readNumber(figure.dataset.seed, 19))

  return {
    n,
    samples: makeMixtureSamples(n, seed),
    seed,
  }
}

function cloneDistributionLearningState(
  state: DistributionLearningState,
): DistributionLearningState {
  return {
    ...state,
    samples: [...state.samples],
  }
}

function regenerateSamples(state: DistributionLearningState) {
  state.samples = makeMixtureSamples(state.n, state.seed)
}

function silvermanBandwidth(samples: number[]) {
  const sampleMean = mean(samples)
  const sampleStd = standardDeviation(samples, sampleMean)
  return Math.max(0.16, 1.06 * sampleStd * samples.length ** -0.2)
}

function estimatedDensity(x: number, samples: number[], bandwidth: number) {
  const total = samples.reduce((sum, sample) => sum + normalPdf(x, sample, bandwidth), 0)
  return total / samples.length
}

function syncDistributionLearningControls(figure: HTMLElement, state: DistributionLearningState) {
  const slider = figure.querySelector<HTMLInputElement>(".distribution-learning-n-slider")
  const output = figure.querySelector<HTMLOutputElement>(".distribution-learning-n")

  if (slider) slider.value = String(sampleSizeToSliderValue(state.n))
  if (output) output.value = String(state.n)

  figure.dataset.n = String(state.n)
  figure.dataset.seed = String(state.seed)
}

function distributionLearningTemplate({
  canExpand,
  id,
  state,
}: {
  canExpand: boolean
  id: string
  state: DistributionLearningState
}) {
  return `
    ${toolbarHtml("Learning a distribution from data", canExpand)}
    <div class="interactive-figure-plot distribution-learning-plot"></div>
    <div class="distribution-learning-readout"></div>
    <label class="gaussian-histogram-slider distribution-learning-slider">
      <div class="gaussian-histogram-slider-row">
        <span>n</span>
        <input class="distribution-learning-n-slider" type="range" min="0" max="${sampleSliderMax}" step="1" value="${sampleSizeToSliderValue(
          state.n,
        )}" aria-describedby="${id}-ticks" />
        <output class="distribution-learning-n">${state.n}</output>
      </div>
      ${sampleTicksHtml(`${id}-ticks`)}
    </label>
  `
}

function drawPanelFrame(
  chart: Selection<SVGGElement, unknown, null, undefined>,
  panel: Panel,
  title: string,
  subtitle: string,
) {
  const group = chart.append("g").attr("transform", `translate(${panel.left},${panel.top})`)

  group
    .append("rect")
    .attr("class", "distribution-learning-panel-bg")
    .attr("width", panel.width)
    .attr("height", panel.height)
    .attr("rx", 8)

  group
    .append("text")
    .attr("class", "distribution-learning-panel-title")
    .attr("x", 14)
    .attr("y", 24)
    .text(title)

  group
    .append("text")
    .attr("class", "distribution-learning-panel-subtitle")
    .attr("x", 14)
    .attr("y", 43)
    .text(subtitle)

  return group
}

function drawDensityPanel(
  group: Selection<SVGGElement, unknown, null, undefined>,
  panel: Panel,
  points: [number, number][],
  yMax: number,
  className: string,
  showGhost = false,
) {
  const plotTop = 56
  const plotBottom = 28
  const plotLeft = 24
  const plotRight = 16
  const plotHeight = panel.height - plotTop - plotBottom
  const plotWidth = panel.width - plotLeft - plotRight
  const xScale = scaleLinear().domain(domain).range([0, plotWidth])
  const yScale = scaleLinear().domain([0, yMax]).range([plotHeight, 0])

  const area = d3Area<[number, number]>()
    .x((d) => xScale(d[0]))
    .y0(yScale(0))
    .y1((d) => yScale(d[1]))
    .curve(curveBasis)
  const line = d3Line<[number, number]>()
    .x((d) => xScale(d[0]))
    .y((d) => yScale(d[1]))
    .curve(curveBasis)
  const plot = group.append("g").attr("transform", `translate(${plotLeft},${plotTop})`)

  plot
    .append("path")
    .datum(points)
    .attr("class", `distribution-learning-density-area ${className}`)
    .attr("d", area)

  plot
    .append("path")
    .datum(points)
    .attr("class", `distribution-learning-density-line ${className}`)
    .attr("d", line)

  if (showGhost) {
    const truePoints = range(0, 160).map((i) => {
      const x = domain[0] + (i * (domain[1] - domain[0])) / 159
      return [x, trueDensity(x)] as [number, number]
    })

    plot
      .append("path")
      .datum(truePoints)
      .attr("class", "distribution-learning-density-line is-ghost")
      .attr("d", line)
  }

  plot
    .append("g")
    .attr("class", "distribution-learning-axis")
    .attr("transform", `translate(0,${plotHeight})`)
    .call(
      axisBottom(xScale)
        .ticks(panel.width < 210 ? 3 : 5)
        .tickSizeOuter(0),
    )

  return { plotHeight, plotLeft, plotTop, plotWidth, xScale, yScale }
}

function drawSamplesPanel(
  group: Selection<SVGGElement, unknown, null, undefined>,
  panel: Panel,
  samples: number[],
  compact: boolean,
) {
  const plotTop = 58
  const plotBottom = 28
  const plotLeft = 24
  const plotRight = 16
  const plotHeight = panel.height - plotTop - plotBottom
  const plotWidth = panel.width - plotLeft - plotRight
  const xScale = scaleLinear().domain(domain).range([0, plotWidth])
  const histogram = d3Bin<number, number>()
    .domain(domain)
    .thresholds(xScale.ticks(compact ? 12 : 16))
    .value((d) => d)
  const bins = histogram(samples)
  const maxCount = max(bins, (bin) => bin.length) ?? 1
  const yScale = scaleLinear().domain([0, maxCount]).range([plotHeight, 8]).nice()
  const plot = group.append("g").attr("transform", `translate(${plotLeft},${plotTop})`)

  plot
    .append("line")
    .attr("class", "distribution-learning-sample-axis-line")
    .attr("x1", 0)
    .attr("x2", plotWidth)
    .attr("y1", plotHeight)
    .attr("y2", plotHeight)

  plot
    .selectAll("rect")
    .data(bins)
    .join("rect")
    .attr("class", "distribution-learning-sample-bucket")
    .attr("x", (bin) => xScale(bin.x0 ?? domain[0]) + 1)
    .attr("y", (bin) => yScale(bin.length))
    .attr("width", (bin) =>
      Math.max(1, xScale(bin.x1 ?? domain[1]) - xScale(bin.x0 ?? domain[0]) - 2),
    )
    .attr("height", (bin) => plotHeight - yScale(bin.length))
    .attr("rx", 2)

  plot
    .append("text")
    .attr("class", "distribution-learning-sample-note")
    .attr("x", plotWidth / 2)
    .attr("y", 12)
    .text("bucketed observations")

  plot
    .append("g")
    .attr("class", "distribution-learning-axis")
    .attr("transform", `translate(0,${plotHeight})`)
    .call(
      axisBottom(xScale)
        .ticks(panel.width < 210 ? 3 : 5)
        .tickSizeOuter(0),
    )
}

function drawArrow(
  chart: Selection<SVGGElement, unknown, null, undefined>,
  start: [number, number],
  end: [number, number],
  label: string,
  compact: boolean,
  markerId: string,
) {
  const midX = (start[0] + end[0]) / 2
  const midY = (start[1] + end[1]) / 2

  chart
    .append("path")
    .attr("class", "distribution-learning-arrow")
    .attr(
      "d",
      compact
        ? `M${start[0]},${start[1]} C${start[0]},${midY} ${end[0]},${midY} ${end[0]},${end[1]}`
        : `M${start[0]},${start[1]} L${end[0]},${end[1]}`,
    )
    .attr("marker-end", `url(#${markerId})`)

  chart
    .append("text")
    .attr("class", "distribution-learning-arrow-label")
    .attr("x", midX)
    .attr("y", compact ? midY - 8 : midY - 10)
    .text(label)
}

function renderDistributionLearning(figure: HTMLElement, state: DistributionLearningState) {
  const plot = figure.querySelector<HTMLElement>(".distribution-learning-plot")
  const readout = figure.querySelector<HTMLElement>(".distribution-learning-readout")
  if (!plot || !readout) return

  const width = Math.max(plot.clientWidth, 280)
  const expanded = figure.classList.contains("is-expanded")
  const compact = width < 680
  const margin = compact
    ? { top: 12, right: 10, bottom: 12, left: 10 }
    : { top: 14, right: 14, bottom: 14, left: 14 }
  const gap = compact ? 28 : 18
  const innerWidth = width - margin.left - margin.right
  const panelHeight = compact ? (expanded ? 170 : 146) : expanded ? 250 : 204
  const panelWidth = compact ? innerWidth : (innerWidth - 2 * gap) / 3
  const height = compact
    ? margin.top + panelHeight * 3 + gap * 2 + margin.bottom
    : margin.top + panelHeight + margin.bottom
  const bandwidth = silvermanBandwidth(state.samples)
  const curveX = range(0, 180).map((i) => domain[0] + (i * (domain[1] - domain[0])) / 179)
  const truePoints = curveX.map((x) => [x, trueDensity(x)] as [number, number])
  const estimatePoints = curveX.map(
    (x) => [x, estimatedDensity(x, state.samples, bandwidth)] as [number, number],
  )
  const yMax =
    Math.max(max(truePoints, (d) => d[1]) ?? 0, max(estimatePoints, (d) => d[1]) ?? 0) * 1.18
  const panels: [Panel, Panel, Panel] = compact
    ? [
        { height: panelHeight, left: margin.left, top: margin.top, width: panelWidth },
        {
          height: panelHeight,
          left: margin.left,
          top: margin.top + panelHeight + gap,
          width: panelWidth,
        },
        {
          height: panelHeight,
          left: margin.left,
          top: margin.top + 2 * (panelHeight + gap),
          width: panelWidth,
        },
      ]
    : [
        { height: panelHeight, left: margin.left, top: margin.top, width: panelWidth },
        {
          height: panelHeight,
          left: margin.left + panelWidth + gap,
          top: margin.top,
          width: panelWidth,
        },
        {
          height: panelHeight,
          left: margin.left + 2 * (panelWidth + gap),
          top: margin.top,
          width: panelWidth,
        },
      ]

  plot.replaceChildren()
  readout.textContent = `The samples are observed; the source distribution is not. n=${state.n}, bandwidth=${bandwidth.toFixed(2)}`
  const markerId = `distribution-learning-arrowhead-${++markerCount}`

  const svg = select(plot)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img")
    .attr(
      "aria-label",
      "Diagram showing an unknown source distribution producing samples, then a learned density estimate from those samples.",
    )
  const chart = svg.append("g")

  const defs = svg.append("defs")
  defs
    .append("marker")
    .attr("id", markerId)
    .attr("viewBox", "0 0 10 10")
    .attr("refX", 8)
    .attr("refY", 5)
    .attr("markerWidth", 7)
    .attr("markerHeight", 7)
    .attr("orient", "auto-start-reverse")
    .append("path")
    .attr("class", "distribution-learning-arrow-marker")
    .attr("d", "M0,0 L10,5 L0,10 Z")

  const sourcePanel = drawPanelFrame(chart, panels[0], "Hidden source", "P* generates data")
  const samplePanel = drawPanelFrame(chart, panels[1], "Observed samples", "dataset D")
  const estimatePanel = drawPanelFrame(chart, panels[2], "Learned estimate", "P-hat from D")

  drawDensityPanel(sourcePanel, panels[0], truePoints, yMax, "is-source")
  drawSamplesPanel(samplePanel, panels[1], state.samples, compact)
  drawDensityPanel(estimatePanel, panels[2], estimatePoints, yMax, "is-estimate", true)

  if (compact) {
    drawArrow(
      chart,
      [panels[0].left + panels[0].width / 2, panels[0].top + panels[0].height + 5],
      [panels[1].left + panels[1].width / 2, panels[1].top - 5],
      "sample",
      compact,
      markerId,
    )
    drawArrow(
      chart,
      [panels[1].left + panels[1].width / 2, panels[1].top + panels[1].height + 5],
      [panels[2].left + panels[2].width / 2, panels[2].top - 5],
      "learn",
      compact,
      markerId,
    )
  } else {
    drawArrow(
      chart,
      [panels[0].left + panels[0].width + 2, panels[0].top + panels[0].height * 0.5],
      [panels[1].left - 5, panels[1].top + panels[1].height * 0.5],
      "sample",
      compact,
      markerId,
    )
    drawArrow(
      chart,
      [panels[1].left + panels[1].width + 2, panels[1].top + panels[1].height * 0.5],
      [panels[2].left - 5, panels[2].top + panels[2].height * 0.5],
      "learn",
      compact,
      markerId,
    )
  }
}

export const distributionLearningDiagramFigure: InteractiveFigureDefinition<DistributionLearningState> =
  {
    bindControls: ({ addCleanup, figure, render, state }) => {
      const slider = figure.querySelector<HTMLInputElement>(".distribution-learning-n-slider")
      const updateSampleSize = () => {
        if (!slider) return

        state.n = sliderValueToSampleSize(Number(slider.value))
        regenerateSamples(state)
        render()
      }

      slider?.addEventListener("input", updateSampleSize)
      addCleanup(() => slider?.removeEventListener("input", updateSampleSize))
    },
    classNames: ["distribution-learning-figure"],
    cloneState: cloneDistributionLearningState,
    readState: readDistributionLearningState,
    render: renderDistributionLearning,
    resample: (state) => {
      state.seed += 1
      regenerateSamples(state)
    },
    serializeState: (state) => ({
      n: state.n,
      seed: state.seed,
    }),
    syncControls: syncDistributionLearningControls,
    template: distributionLearningTemplate,
    type: "distribution-learning-diagram",
  }

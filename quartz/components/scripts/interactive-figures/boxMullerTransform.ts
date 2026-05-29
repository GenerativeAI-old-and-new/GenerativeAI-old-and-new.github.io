import {
  axisBottom,
  bin as d3Bin,
  curveMonotoneX,
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
  normalPdf,
  readNumber,
  sampleSizeToSliderValue,
  sampleSliderMax,
  seededRandom,
  sliderValueToSampleSize,
  sortedQuantile,
} from "./math"
import { sampleTicksHtml, toolbarHtml } from "./ui"

type BoxMullerPoint = {
  r: number
  theta: number
  u1: number
  u2: number
  z1: number
  z2: number
}

type BoxMullerState = {
  binCount: number
  focusIndex: number
  n: number
  points: BoxMullerPoint[]
  seed: number
}

type Rect = {
  h: number
  w: number
  x: number
  y: number
}

type SvgSelection = Selection<SVGSVGElement, unknown, null, undefined>

function cssColor(element: HTMLElement, name: string, fallback: string) {
  const color = getComputedStyle(element).getPropertyValue(name).trim()
  return color || fallback
}

function clampFocusIndex(value: number, n: number) {
  return Math.min(Math.max(0, Math.round(value)), Math.max(0, n - 1))
}

function makeBoxMullerPoints(n: number, seed: number) {
  const random = seededRandom(seed)
  const points: BoxMullerPoint[] = []

  for (let i = 0; i < n; i++) {
    const u1 = Math.max(random(), Number.EPSILON)
    const u2 = random()
    const r = Math.sqrt(-2 * Math.log(u1))
    const theta = 2 * Math.PI * u2

    points.push({
      r,
      theta,
      u1,
      u2,
      z1: r * Math.cos(theta),
      z2: r * Math.sin(theta),
    })
  }

  return points
}

function readBoxMullerState(figure: HTMLElement): BoxMullerState {
  const n = clampSampleSize(readNumber(figure.dataset.n, 800))
  const focusIndex = clampFocusIndex(readNumber(figure.dataset.focusIndex, 0), n)
  const state = {
    binCount: Math.round(readNumber(figure.dataset.bins, 24)),
    focusIndex,
    n,
    points: [] as BoxMullerPoint[],
    seed: Math.round(readNumber(figure.dataset.seed, 19)),
  }

  state.points = makeBoxMullerPoints(state.n, state.seed)
  return state
}

function cloneBoxMullerState(state: BoxMullerState): BoxMullerState {
  return {
    ...state,
    points: state.points.map((point) => ({ ...point })),
  }
}

function regenerateBoxMullerPoints(state: BoxMullerState) {
  state.points = makeBoxMullerPoints(state.n, state.seed)
  state.focusIndex = clampFocusIndex(state.focusIndex, state.n)
}

function syncBoxMullerControls(figure: HTMLElement, state: BoxMullerState) {
  const nSlider = figure.querySelector<HTMLInputElement>(".box-muller-n-slider")
  const focusSlider = figure.querySelector<HTMLInputElement>(".box-muller-focus-slider")
  const nOutput = figure.querySelector<HTMLOutputElement>(".box-muller-n")
  const focusOutput = figure.querySelector<HTMLOutputElement>(".box-muller-focus")

  if (nSlider) nSlider.value = String(sampleSizeToSliderValue(state.n))
  if (focusSlider) {
    focusSlider.max = String(Math.max(0, state.n - 1))
    focusSlider.value = String(state.focusIndex)
  }
  if (nOutput) nOutput.value = String(state.n)
  if (focusOutput) focusOutput.value = String(state.focusIndex + 1)

  figure.dataset.n = String(state.n)
  figure.dataset.focusIndex = String(state.focusIndex)
  figure.dataset.seed = String(state.seed)
}

function boxMullerTemplate({
  canExpand,
  id,
  state,
}: {
  canExpand: boolean
  id: string
  state: BoxMullerState
}) {
  return `
    ${toolbarHtml("Box-Muller transform", canExpand)}
    <div class="interactive-figure-plot box-muller-plot"></div>
    <div class="box-muller-stats"></div>
    <div class="box-muller-controls">
      <label class="gaussian-histogram-slider">
        <div class="gaussian-histogram-slider-row">
          <span>n</span>
          <input class="box-muller-n-slider" type="range" min="0" max="${sampleSliderMax}" step="1" value="${sampleSizeToSliderValue(
            state.n,
          )}" aria-describedby="${id}-n-ticks" />
          <output class="box-muller-n">${state.n}</output>
        </div>
        ${sampleTicksHtml(`${id}-n-ticks`)}
      </label>
      <label class="gaussian-histogram-slider">
        <div class="gaussian-histogram-slider-row">
          <span>pair</span>
          <input class="box-muller-focus-slider" type="range" min="0" max="${Math.max(
            0,
            state.n - 1,
          )}" step="1" value="${state.focusIndex}" />
          <output class="box-muller-focus">${state.focusIndex + 1}</output>
        </div>
      </label>
    </div>
  `
}

function drawPanelLabel(svg: SvgSelection, text: string, rect: Rect) {
  svg
    .append("text")
    .attr("class", "box-muller-panel-label")
    .attr("x", rect.x)
    .attr("y", rect.y - 12)
    .text(text)
}

function drawArrow(svg: SvgSelection, startX: number, startY: number, endX: number) {
  svg
    .append("path")
    .attr("class", "box-muller-arrow")
    .attr(
      "d",
      `M${startX},${startY} C${startX + 18},${startY} ${endX - 18},${startY} ${endX},${startY}`,
    )
    .attr("marker-end", "url(#box-muller-arrowhead)")
}

function drawSquarePanel(
  svg: SvgSelection,
  rect: Rect,
  points: BoxMullerPoint[],
  focus: BoxMullerPoint,
  colors: ReturnType<typeof figureColors>,
  expanded: boolean,
) {
  const xScale = scaleLinear()
    .domain([0, 1])
    .range([rect.x, rect.x + rect.w])
  const yScale = scaleLinear()
    .domain([0, 1])
    .range([rect.y + rect.h, rect.y])
  const visible = points.slice(0, expanded ? 420 : 240)

  drawPanelLabel(svg, "Uniform inputs", rect)

  svg
    .append("rect")
    .attr("class", "box-muller-panel")
    .attr("x", rect.x)
    .attr("y", rect.y)
    .attr("width", rect.w)
    .attr("height", rect.h)

  svg
    .append("g")
    .selectAll("circle")
    .data(visible)
    .join("circle")
    .attr("class", "box-muller-sample")
    .attr("cx", (d) => xScale(d.u1))
    .attr("cy", (d) => yScale(d.u2))
    .attr("r", expanded ? 2.4 : 1.9)
    .attr("fill", colors.hist)

  svg
    .append("circle")
    .attr("class", "box-muller-focus-point")
    .attr("cx", xScale(focus.u1))
    .attr("cy", yScale(focus.u2))
    .attr("r", expanded ? 5.6 : 4.7)
    .attr("fill", colors.trueLine)

  svg
    .append("text")
    .attr("class", "box-muller-axis-label")
    .attr("x", rect.x + rect.w / 2)
    .attr("y", rect.y + rect.h + 24)
    .attr("text-anchor", "middle")
    .text("U1")

  svg
    .append("text")
    .attr("class", "box-muller-axis-label")
    .attr("x", rect.x - 20)
    .attr("y", rect.y + rect.h / 2)
    .attr("text-anchor", "middle")
    .attr("transform", `rotate(-90 ${rect.x - 20} ${rect.y + rect.h / 2})`)
    .text("U2")
}

function drawPlanePanel(
  svg: SvgSelection,
  rect: Rect,
  points: BoxMullerPoint[],
  focus: BoxMullerPoint,
  colors: ReturnType<typeof figureColors>,
  expanded: boolean,
) {
  const absValues = points.flatMap((point) => [Math.abs(point.z1), Math.abs(point.z2)])
  const limit = Math.min(4.4, Math.max(3.1, sortedQuantile(absValues, 0.985)))
  const xScale = scaleLinear()
    .domain([-limit, limit])
    .range([rect.x, rect.x + rect.w])
  const yScale = scaleLinear()
    .domain([-limit, limit])
    .range([rect.y + rect.h, rect.y])
  const visible = points
    .filter((point) => Math.abs(point.z1) <= limit && Math.abs(point.z2) <= limit)
    .slice(0, expanded ? 620 : 360)
  const center = { x: xScale(0), y: yScale(0) }
  const focusX = xScale(Math.max(-limit, Math.min(limit, focus.z1)))
  const focusY = yScale(Math.max(-limit, Math.min(limit, focus.z2)))

  drawPanelLabel(svg, "Gaussian plane", rect)

  svg
    .append("rect")
    .attr("class", "box-muller-panel")
    .attr("x", rect.x)
    .attr("y", rect.y)
    .attr("width", rect.w)
    .attr("height", rect.h)

  const plane = svg.append("g")

  for (const radius of [1, 2, 3]) {
    plane
      .append("circle")
      .attr("class", "box-muller-plane-ring")
      .attr("cx", center.x)
      .attr("cy", center.y)
      .attr("r", Math.abs(xScale(radius) - xScale(0)))
  }

  plane
    .append("line")
    .attr("class", "box-muller-plane-axis")
    .attr("x1", rect.x)
    .attr("x2", rect.x + rect.w)
    .attr("y1", center.y)
    .attr("y2", center.y)

  plane
    .append("line")
    .attr("class", "box-muller-plane-axis")
    .attr("x1", center.x)
    .attr("x2", center.x)
    .attr("y1", rect.y)
    .attr("y2", rect.y + rect.h)

  plane
    .append("g")
    .selectAll("circle")
    .data(visible)
    .join("circle")
    .attr("class", "box-muller-sample")
    .attr("cx", (d) => xScale(d.z1))
    .attr("cy", (d) => yScale(d.z2))
    .attr("r", expanded ? 2.2 : 1.7)
    .attr("fill", colors.histStroke)

  plane
    .append("line")
    .attr("class", "box-muller-radius-line")
    .attr("x1", center.x)
    .attr("y1", center.y)
    .attr("x2", focusX)
    .attr("y2", focusY)

  const angleRadius = Math.min(rect.w, rect.h) * 0.18
  const anglePoints = range(0, 28).map((i) => {
    const theta = (focus.theta * i) / 27
    return [center.x + angleRadius * Math.cos(theta), center.y - angleRadius * Math.sin(theta)] as [
      number,
      number,
    ]
  })
  const angleLine = d3Line<[number, number]>()
    .x((d) => d[0])
    .y((d) => d[1])

  plane.append("path").datum(anglePoints).attr("class", "box-muller-angle-arc").attr("d", angleLine)

  plane
    .append("circle")
    .attr("class", "box-muller-focus-point")
    .attr("cx", focusX)
    .attr("cy", focusY)
    .attr("r", expanded ? 5.8 : 4.9)
    .attr("fill", colors.trueLine)

  plane
    .append("text")
    .attr("class", "box-muller-axis-label")
    .attr("x", rect.x + rect.w - 6)
    .attr("y", center.y - 7)
    .attr("text-anchor", "end")
    .text("Z1")

  plane
    .append("text")
    .attr("class", "box-muller-axis-label")
    .attr("x", center.x + 7)
    .attr("y", rect.y + 13)
    .text("Z2")
}

function drawHistogramPanel(
  svg: SvgSelection,
  rect: Rect,
  values: number[],
  title: string,
  colors: ReturnType<typeof figureColors>,
  binCount: number,
) {
  const margin = { top: 18, right: 10, bottom: 25, left: 26 }
  const innerWidth = rect.w - margin.left - margin.right
  const innerHeight = rect.h - margin.top - margin.bottom
  const xScale = scaleLinear().domain([-3.6, 3.6]).range([0, innerWidth])
  const bins = d3Bin<number, number>()
    .domain(xScale.domain() as [number, number])
    .thresholds(xScale.ticks(binCount))
    .value((d) => d)(values)
  const maxDensity = max(bins, (d) => {
    const width = (d.x1 ?? 0) - (d.x0 ?? 0)
    return width > 0 ? d.length / (values.length * width) : 0
  })
  const yScale = scaleLinear()
    .domain([0, Math.max(maxDensity ?? 0, normalPdf(0, 0, 1)) * 1.18])
    .range([innerHeight, 0])
  const curveX = range(0, 96).map((i) => -3.6 + (7.2 * i) / 95)
  const densityLine = d3Line<[number, number]>()
    .x((d) => xScale(d[0]))
    .y((d) => yScale(d[1]))
    .curve(curveMonotoneX)
  const panel = svg.append("g").attr("transform", `translate(${rect.x},${rect.y})`)
  const chart = panel.append("g").attr("transform", `translate(${margin.left},${margin.top})`)

  panel
    .append("rect")
    .attr("class", "box-muller-panel")
    .attr("width", rect.w)
    .attr("height", rect.h)

  panel
    .append("text")
    .attr("class", "box-muller-mini-label")
    .attr("x", margin.left)
    .attr("y", 13)
    .text(title)

  chart
    .selectAll("rect")
    .data(bins)
    .join("rect")
    .attr("class", "box-muller-hist-bar")
    .attr("x", (d) => xScale(d.x0 ?? 0) + 1)
    .attr("y", (d) => {
      const width = (d.x1 ?? 0) - (d.x0 ?? 0)
      const density = width > 0 ? d.length / (values.length * width) : 0
      return yScale(density)
    })
    .attr("width", (d) => Math.max(0, xScale(d.x1 ?? 0) - xScale(d.x0 ?? 0) - 1))
    .attr("height", (d) => {
      const width = (d.x1 ?? 0) - (d.x0 ?? 0)
      const density = width > 0 ? d.length / (values.length * width) : 0
      return innerHeight - yScale(density)
    })
    .attr("fill", colors.hist)
    .attr("stroke", colors.histStroke)

  chart
    .append("path")
    .datum(curveX.map((x) => [x, normalPdf(x, 0, 1)] as [number, number]))
    .attr("class", "box-muller-density-line")
    .attr("d", densityLine)
    .attr("fill", "none")
    .attr("stroke", colors.trueLine)

  chart
    .append("g")
    .attr("class", "box-muller-axis")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(
      axisBottom(xScale)
        .ticks(rect.w < 180 ? 3 : 5)
        .tickSizeOuter(0),
    )
}

function figureColors(figure: HTMLElement) {
  return {
    axis: cssColor(figure, "--chart-axis", "#726a60"),
    grid: cssColor(figure, "--chart-grid", "#e7e0d6"),
    hist: cssColor(figure, "--chart-hist", "#a9cdec"),
    histStroke: cssColor(figure, "--chart-hist-stroke", "#2c5f9e"),
    trueLine: cssColor(figure, "--chart-true", "#9f3d46"),
  }
}

function renderBoxMuller(figure: HTMLElement, state: BoxMullerState) {
  const plot = figure.querySelector<HTMLElement>(".box-muller-plot")
  const stats = figure.querySelector<HTMLElement>(".box-muller-stats")
  if (!plot || !stats) return

  const width = Math.max(plot.clientWidth, 320)
  const expanded = figure.classList.contains("is-expanded")
  const compact = width < 680
  const layout = figure.dataset.figureLayout
  const height = compact
    ? expanded
      ? 960
      : 900
    : expanded
      ? Math.min(Math.max(width * 0.48, 460), 580)
      : layout === "modal"
        ? Math.min(Math.max(width * 0.42, 420), 520)
        : Math.min(Math.max(width * 0.34, 300), 360)
  const colors = figureColors(figure)
  const focus = state.points[state.focusIndex] ?? state.points[0]
  const z1Values = state.points.map((point) => point.z1)
  const z2Values = state.points.map((point) => point.z2)

  plot.replaceChildren()
  stats.textContent = `pair ${state.focusIndex + 1}/${state.n}: U1=${focus.u1.toFixed(
    3,
  )}, U2=${focus.u2.toFixed(3)} -> R=${focus.r.toFixed(3)}, θ=${focus.theta.toFixed(
    3,
  )} -> (Z1, Z2)=(${focus.z1.toFixed(3)}, ${focus.z2.toFixed(3)})`

  const svg = select(plot)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img")
    .attr(
      "aria-label",
      "Box-Muller transform from two uniform random variables to two standard normal random variables",
    )

  svg
    .append("defs")
    .append("marker")
    .attr("id", "box-muller-arrowhead")
    .attr("viewBox", "0 0 10 10")
    .attr("refX", 8)
    .attr("refY", 5)
    .attr("markerWidth", 5)
    .attr("markerHeight", 5)
    .attr("orient", "auto-start-reverse")
    .append("path")
    .attr("d", "M 0 0 L 10 5 L 0 10 z")
    .attr("fill", colors.axis)

  svg
    .append("text")
    .attr("class", "box-muller-formula")
    .attr("x", 32)
    .attr("y", 24)
    .text("R = sqrt(-2 log U1),  θ = 2πU2")
  svg
    .append("text")
    .attr("class", "box-muller-formula")
    .attr("x", 32)
    .attr("y", 42)
    .text("Z1 = R cos θ,  Z2 = R sin θ")

  if (compact) {
    const panelWidth = Math.min(width - 64, 380)
    const squareSize = Math.min(panelWidth, expanded ? 220 : 190)
    const square: Rect = { h: squareSize, w: squareSize, x: 40, y: 76 }
    const plane: Rect = { h: squareSize, w: squareSize, x: 40, y: square.y + square.h + 48 }
    const hist: Rect = {
      h: expanded ? 140 : 116,
      w: width - 80,
      x: 40,
      y: plane.y + plane.h + 50,
    }

    drawSquarePanel(svg, square, state.points, focus, colors, expanded)
    drawPlanePanel(svg, plane, state.points, focus, colors, expanded)
    drawHistogramPanel(svg, hist, z1Values, "Z1 histogram", colors, state.binCount)
    drawHistogramPanel(
      svg,
      { ...hist, y: hist.y + hist.h + 22 },
      z2Values,
      "Z2 histogram",
      colors,
      state.binCount,
    )
  } else {
    const marginX = 28
    const gap = 24
    const panelTop = 76
    const panelHeight = height - panelTop - 24
    const squareSize = Math.min(panelHeight, Math.max(150, width * 0.18))
    const planeSize = Math.min(panelHeight, Math.max(190, width * 0.24))
    const histWidth = width - 2 * marginX - squareSize - planeSize - 2 * gap
    const square: Rect = {
      h: squareSize,
      w: squareSize,
      x: marginX,
      y: panelTop + (panelHeight - squareSize) / 2,
    }
    const plane: Rect = {
      h: planeSize,
      w: planeSize,
      x: square.x + square.w + gap,
      y: panelTop + (panelHeight - planeSize) / 2,
    }
    const histHeight = (panelHeight - 24) / 2
    const histX = plane.x + plane.w + gap
    const hist1: Rect = { h: histHeight, w: histWidth, x: histX, y: panelTop }
    const hist2: Rect = { h: histHeight, w: histWidth, x: histX, y: panelTop + histHeight + 24 }

    drawSquarePanel(svg, square, state.points, focus, colors, expanded)
    drawArrow(svg, square.x + square.w + 8, square.y + square.h / 2, plane.x - 10)
    drawPlanePanel(svg, plane, state.points, focus, colors, expanded)
    drawArrow(svg, plane.x + plane.w + 8, plane.y + plane.h / 2, hist1.x - 10)
    drawPanelLabel(svg, "Marginal checks", hist1)
    drawHistogramPanel(svg, hist1, z1Values, "Z1 histogram", colors, state.binCount)
    drawHistogramPanel(svg, hist2, z2Values, "Z2 histogram", colors, state.binCount)
  }

  svg.selectAll(".box-muller-axis path, .box-muller-axis line").attr("stroke", colors.axis)
  svg.selectAll(".box-muller-axis text").attr("fill", colors.axis)
  svg.selectAll(".box-muller-plane-axis").attr("stroke", colors.axis)
  svg.selectAll(".box-muller-plane-ring").attr("stroke", colors.grid)
  svg.selectAll(".box-muller-arrow").attr("stroke", colors.axis)
}

export const boxMullerTransformFigure: InteractiveFigureDefinition<BoxMullerState> = {
  bindControls: ({ addCleanup, figure, render, state }) => {
    const nSlider = figure.querySelector<HTMLInputElement>(".box-muller-n-slider")
    const focusSlider = figure.querySelector<HTMLInputElement>(".box-muller-focus-slider")
    const updateSampleSize = () => {
      if (!nSlider) return

      state.n = sliderValueToSampleSize(Number(nSlider.value))
      state.focusIndex = clampFocusIndex(state.focusIndex, state.n)
      regenerateBoxMullerPoints(state)
      render()
    }
    const updateFocus = () => {
      if (!focusSlider) return

      state.focusIndex = clampFocusIndex(Number(focusSlider.value), state.n)
      render()
    }

    nSlider?.addEventListener("input", updateSampleSize)
    focusSlider?.addEventListener("input", updateFocus)
    addCleanup(() => {
      nSlider?.removeEventListener("input", updateSampleSize)
      focusSlider?.removeEventListener("input", updateFocus)
    })
  },
  classNames: ["box-muller-transform-figure"],
  cloneState: cloneBoxMullerState,
  readState: readBoxMullerState,
  render: renderBoxMuller,
  resample: (state) => {
    state.seed += 1
    regenerateBoxMullerPoints(state)
  },
  serializeState: (state) => ({
    bins: state.binCount,
    focusIndex: state.focusIndex,
    n: state.n,
    seed: state.seed,
  }),
  syncControls: syncBoxMullerControls,
  template: boxMullerTemplate,
  type: "box-muller-transform",
}

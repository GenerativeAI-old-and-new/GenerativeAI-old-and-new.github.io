import { line as d3Line, range, scaleLinear, select, type Selection } from "d3"
import { InteractiveFigureDefinition } from "./core"
import {
  clampSampleSize,
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

function drawArrow(svg: SvgSelection, startX: number, startY: number, endX: number, endY: number) {
  svg
    .append("path")
    .attr("class", "box-muller-arrow")
    .attr(
      "d",
      `M${startX},${startY} C${startX + 18},${startY} ${endX - 18},${endY} ${endX},${endY}`,
    )
    .attr("marker-end", "url(#box-muller-arrowhead)")
}

function drawAnglePanel(
  svg: SvgSelection,
  rect: Rect,
  focus: BoxMullerPoint,
  colors: ReturnType<typeof figureColors>,
) {
  const center = { x: rect.x + rect.w / 2, y: rect.y + rect.h / 2 }
  const radius = Math.min(rect.w, rect.h) * 0.32
  const focusX = center.x + radius * Math.cos(focus.theta)
  const focusY = center.y - radius * Math.sin(focus.theta)
  const arcRadius = radius * 0.42
  const anglePoints = range(0, 24).map((i) => {
    const theta = (focus.theta * i) / 23
    return [center.x + arcRadius * Math.cos(theta), center.y - arcRadius * Math.sin(theta)] as [
      number,
      number,
    ]
  })
  const angleLine = d3Line<[number, number]>()
    .x((d) => d[0])
    .y((d) => d[1])

  drawPanelLabel(svg, "Angle", rect)

  svg
    .append("rect")
    .attr("class", "box-muller-panel")
    .attr("x", rect.x)
    .attr("y", rect.y)
    .attr("width", rect.w)
    .attr("height", rect.h)

  svg
    .append("circle")
    .attr("class", "box-muller-wheel")
    .attr("cx", center.x)
    .attr("cy", center.y)
    .attr("r", radius)

  svg
    .append("line")
    .attr("class", "box-muller-plane-axis")
    .attr("x1", center.x - radius)
    .attr("x2", center.x + radius)
    .attr("y1", center.y)
    .attr("y2", center.y)

  svg
    .append("line")
    .attr("class", "box-muller-plane-axis")
    .attr("x1", center.x)
    .attr("x2", center.x)
    .attr("y1", center.y + radius)
    .attr("y2", center.y - radius)

  svg
    .append("line")
    .attr("class", "box-muller-radius-line")
    .attr("x1", center.x)
    .attr("y1", center.y)
    .attr("x2", focusX)
    .attr("y2", focusY)

  svg.append("path").datum(anglePoints).attr("class", "box-muller-angle-arc").attr("d", angleLine)

  svg
    .append("circle")
    .attr("class", "box-muller-focus-point")
    .attr("cx", focusX)
    .attr("cy", focusY)
    .attr("r", 4.7)
    .attr("fill", colors.trueLine)

  svg
    .append("text")
    .attr("class", "box-muller-axis-label")
    .attr("x", focusX + 7)
    .attr("y", focusY - 6)
    .text("θ")
}

function drawRadiusPanel(
  svg: SvgSelection,
  rect: Rect,
  focus: BoxMullerPoint,
  colors: ReturnType<typeof figureColors>,
) {
  const margin = { top: 18, right: 14, bottom: 22, left: 26 }
  const innerWidth = rect.w - margin.left - margin.right
  const innerHeight = rect.h - margin.top - margin.bottom
  const xScale = scaleLinear().domain([0, 1]).range([0, innerWidth])
  const yScale = scaleLinear().domain([0, 3.4]).range([innerHeight, 0])
  const curvePoints = range(0, 120).map((i) => {
    const u = 0.004 + (0.996 * i) / 119
    return [u, Math.min(Math.sqrt(-2 * Math.log(u)), 3.4)] as [number, number]
  })
  const curve = d3Line<[number, number]>()
    .x((d) => xScale(d[0]))
    .y((d) => yScale(d[1]))
  const focusU = Math.max(0.004, Math.min(1, focus.u1))
  const focusR = Math.min(focus.r, 3.4)
  const focusX = xScale(focusU)
  const focusY = yScale(focusR)
  const panel = svg.append("g").attr("transform", `translate(${rect.x},${rect.y})`)
  const chart = panel.append("g").attr("transform", `translate(${margin.left},${margin.top})`)

  drawPanelLabel(svg, "Radius", rect)

  panel
    .append("rect")
    .attr("class", "box-muller-panel")
    .attr("width", rect.w)
    .attr("height", rect.h)

  chart
    .append("line")
    .attr("class", "box-muller-plane-axis")
    .attr("x1", 0)
    .attr("x2", innerWidth)
    .attr("y1", innerHeight)
    .attr("y2", innerHeight)

  chart
    .append("line")
    .attr("class", "box-muller-plane-axis")
    .attr("x1", 0)
    .attr("x2", 0)
    .attr("y1", innerHeight)
    .attr("y2", 0)

  chart
    .append("path")
    .datum(curvePoints)
    .attr("class", "box-muller-radius-curve")
    .attr("d", curve)
    .attr("fill", "none")
    .attr("stroke", colors.histStroke)

  chart
    .append("path")
    .attr("class", "box-muller-guide")
    .attr("d", `M${focusX},${innerHeight} V${focusY} H0`)

  chart
    .append("circle")
    .attr("class", "box-muller-focus-point")
    .attr("cx", focusX)
    .attr("cy", focusY)
    .attr("r", 4.5)
    .attr("fill", colors.trueLine)

  panel
    .append("text")
    .attr("class", "box-muller-axis-label")
    .attr("x", margin.left + innerWidth)
    .attr("y", rect.h - 6)
    .attr("text-anchor", "end")
    .text("U1")

  panel
    .append("text")
    .attr("class", "box-muller-axis-label")
    .attr("x", 8)
    .attr("y", margin.top + 8)
    .text("R")
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
      ? 760
      : 680
    : expanded
      ? Math.min(Math.max(width * 0.42, 420), 520)
      : layout === "modal"
        ? Math.min(Math.max(width * 0.36, 370), 460)
        : Math.min(Math.max(width * 0.3, 280), 340)
  const colors = figureColors(figure)
  const focus = state.points[state.focusIndex] ?? state.points[0]

  plot.replaceChildren()
  stats.textContent = `pair ${state.focusIndex + 1}/${state.n}: U1=${focus.u1.toFixed(
    3,
  )} -> R=${focus.r.toFixed(3)} · U2=${focus.u2.toFixed(3)} -> θ=${focus.theta.toFixed(
    3,
  )} · Z=(${focus.z1.toFixed(3)}, ${focus.z2.toFixed(3)})`

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

  if (compact) {
    const panelWidth = Math.min(width - 64, 420)
    const smallHeight = expanded ? 150 : 132
    const planeSize = Math.min(panelWidth, expanded ? 300 : 250)
    const angle: Rect = { h: smallHeight, w: panelWidth, x: (width - panelWidth) / 2, y: 34 }
    const radius: Rect = { ...angle, y: angle.y + angle.h + 38 }
    const plane: Rect = {
      h: planeSize,
      w: planeSize,
      x: (width - planeSize) / 2,
      y: radius.y + radius.h + 46,
    }

    drawAnglePanel(svg, angle, focus, colors)
    drawRadiusPanel(svg, radius, focus, colors)
    drawPlanePanel(svg, plane, state.points, focus, colors, expanded)
    drawArrow(
      svg,
      angle.x + angle.w / 2,
      angle.y + angle.h + 10,
      plane.x + plane.w / 2,
      plane.y - 12,
    )
    drawArrow(
      svg,
      radius.x + radius.w / 2,
      radius.y + radius.h + 10,
      plane.x + plane.w / 2,
      plane.y - 12,
    )
  } else {
    const marginX = 28
    const gap = 24
    const panelTop = 34
    const panelHeight = height - panelTop - 28
    const sideWidth = Math.min(Math.max(width * 0.26, 190), 260)
    const smallHeight = (panelHeight - gap) / 2
    const planeSize = Math.min(panelHeight, width - 2 * marginX - sideWidth - gap)
    const totalWidth = sideWidth + gap + planeSize
    const startX = (width - totalWidth) / 2
    const angle: Rect = { h: smallHeight, w: sideWidth, x: startX, y: panelTop }
    const radius: Rect = { ...angle, y: panelTop + smallHeight + gap }
    const plane: Rect = {
      h: planeSize,
      w: planeSize,
      x: angle.x + angle.w + gap,
      y: panelTop + (panelHeight - planeSize) / 2,
    }

    drawAnglePanel(svg, angle, focus, colors)
    drawRadiusPanel(svg, radius, focus, colors)
    drawArrow(
      svg,
      angle.x + angle.w + 8,
      angle.y + angle.h / 2,
      plane.x - 12,
      plane.y + plane.h * 0.38,
    )
    drawArrow(
      svg,
      radius.x + radius.w + 8,
      radius.y + radius.h / 2,
      plane.x - 12,
      plane.y + plane.h * 0.62,
    )
    drawPlanePanel(svg, plane, state.points, focus, colors, expanded)
  }

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

import {
  axisBottom,
  axisLeft,
  bin as d3Bin,
  curveMonotoneX,
  line as d3Line,
  max,
  scaleLinear,
  select,
} from "d3"

export type DensityCurve = {
  className?: string
  colorVar: string
  dashArray?: string
  fallbackColor: string
  points: [number, number][]
  strokeWidth?: number
}

type DensityHistogramOptions = {
  ariaLabel: string
  binCount: number
  curves: DensityCurve[]
  domain: [number, number]
  figure: HTMLElement
  histogramSamples?: number[]
  samples: number[]
  statsText: string
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

export function renderDensityHistogram({
  ariaLabel,
  binCount,
  curves,
  domain,
  figure,
  histogramSamples = [],
  samples,
  statsText,
}: DensityHistogramOptions) {
  const plot = figure.querySelector<HTMLElement>(".gaussian-histogram-plot")
  const stats = figure.querySelector<HTMLElement>(".gaussian-histogram-stats")
  const tooltip = figure.querySelector<HTMLElement>(".gaussian-histogram-tooltip")
  if (!plot || !stats || !tooltip) return

  const visibleSamples = histogramSamples.length > 0 ? histogramSamples : samples
  const width = Math.max(plot.clientWidth, 224)
  const isExpanded = figure.classList.contains("is-expanded")
  const height = isExpanded ? Math.min(Math.max(width * 0.48, 340), 460) : width < 250 ? 174 : 190
  const margin = isExpanded
    ? { top: 22, right: 22, bottom: 42, left: 50 }
    : { top: 14, right: 12, bottom: 28, left: 34 }
  const innerWidth = width - margin.left - margin.right
  const innerHeight = height - margin.top - margin.bottom

  const xScale = scaleLinear().domain(domain).range([0, innerWidth]).nice()
  const histogram = d3Bin<number, number>()
    .domain(xScale.domain() as [number, number])
    .thresholds(xScale.ticks(binCount))
    .value((d) => d)
  const bins = histogram(visibleSamples)
  const barMax = max(bins, (d) => {
    const width = (d.x1 ?? 0) - (d.x0 ?? 0)
    return width > 0 ? d.length / (samples.length * width) : 0
  })
  const curveMax = max(
    curves.flatMap((curve) => curve.points),
    (d) => d[1],
  )
  const yScale = scaleLinear()
    .domain([0, Math.max(barMax ?? 0, curveMax ?? 0) * 1.16])
    .range([innerHeight, 0])
    .nice()

  const histColor = cssColor(figure, "--chart-hist", "#8dbfe8")
  const histStroke = cssColor(figure, "--chart-hist-stroke", "#2c5f9e")
  const axisColor = cssColor(figure, "--chart-axis", "#6c645a")
  const gridColor = cssColor(figure, "--chart-grid", "#e3ddd2")

  plot.replaceChildren()
  stats.textContent = statsText

  const svg = select(plot)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img")
    .attr("aria-label", ariaLabel)
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
      const density = width > 0 ? d.length / (samples.length * width) : 0
      return yScale(density)
    })
    .attr("width", (d) => Math.max(0, xScale(d.x1 ?? 0) - xScale(d.x0 ?? 0) - 1))
    .attr("height", (d) => {
      const width = (d.x1 ?? 0) - (d.x0 ?? 0)
      const density = width > 0 ? d.length / (samples.length * width) : 0
      return innerHeight - yScale(density)
    })
    .attr("fill", histColor)
    .attr("stroke", histStroke)
    .attr("stroke-width", 0.7)
    .attr("rx", 1.4)
    .on("pointerenter", (event: PointerEvent, d) => {
      const width = (d.x1 ?? 0) - (d.x0 ?? 0)
      const density = width > 0 ? d.length / (samples.length * width) : 0

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

  for (const curve of curves) {
    chart
      .append("path")
      .datum(curve.points)
      .attr("class", curve.className ?? "gaussian-histogram-line")
      .attr("d", densityLine)
      .attr("fill", "none")
      .attr("stroke", cssColor(figure, curve.colorVar, curve.fallbackColor))
      .attr("stroke-linecap", "round")
      .attr("stroke-width", curve.strokeWidth ?? 2.2)
      .attr("stroke-dasharray", curve.dashArray ?? null)
  }

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

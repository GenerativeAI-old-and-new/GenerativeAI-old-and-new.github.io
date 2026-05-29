import { range } from "d3"
import { renderDensityHistogram } from "./densityHistogram"
import { InteractiveFigureDefinition } from "./core"
import {
  clampSampleSize,
  makeGaussianSamples,
  mean,
  normalPdf,
  readNumber,
  sampleExtent,
  sampleSizeToSliderValue,
  sampleSliderMax,
  sliderValueToSampleSize,
  standardDeviation,
} from "./math"
import { sampleTicksHtml, toolbarHtml, tooltipHtml } from "./ui"

type GaussianState = {
  binCount: number
  mu: number
  n: number
  seed: number
  sigma: number
  samples: number[]
}

function readGaussianState(figure: HTMLElement): GaussianState {
  const n = clampSampleSize(readNumber(figure.dataset.n, 1000))
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

function regenerateGaussianSamples(state: GaussianState) {
  state.samples = makeGaussianSamples(state.mu, state.sigma, state.n, state.seed)
}

function syncGaussianControls(figure: HTMLElement, state: GaussianState) {
  const slider = figure.querySelector<HTMLInputElement>(".gaussian-histogram-slider input")
  const output = figure.querySelector<HTMLOutputElement>(".gaussian-histogram-n")

  if (slider) slider.value = String(sampleSizeToSliderValue(state.n))
  if (output) output.value = String(state.n)

  figure.dataset.n = String(state.n)
  figure.dataset.seed = String(state.seed)
}

function gaussianTemplate({
  canExpand,
  id,
  state,
}: {
  canExpand: boolean
  id: string
  state: GaussianState
}) {
  return `
    ${toolbarHtml("Gaussian samples", canExpand)}
    <div class="interactive-figure-plot gaussian-histogram-plot"></div>
    <div class="gaussian-histogram-legend" aria-hidden="true">
      <span><i class="is-true"></i>True</span>
      <span><i class="is-fit"></i>Fit</span>
    </div>
    <div class="gaussian-histogram-stats"></div>
    <label class="gaussian-histogram-slider">
      <div class="gaussian-histogram-slider-row">
        <span>n</span>
        <input type="range" min="0" max="${sampleSliderMax}" step="1" value="${sampleSizeToSliderValue(
          state.n,
        )}" aria-describedby="${id}-ticks" />
        <output class="gaussian-histogram-n">${state.n}</output>
      </div>
      ${sampleTicksHtml(`${id}-ticks`)}
    </label>
    ${tooltipHtml()}
  `
}

function renderGaussian(figure: HTMLElement, state: GaussianState) {
  const sampleMean = mean(state.samples)
  const sampleStd = standardDeviation(state.samples, sampleMean)
  const [sampleMin, sampleMax] = sampleExtent(state.samples)
  const domain: [number, number] = [
    Math.min(state.mu - 4 * state.sigma, sampleMin),
    Math.max(state.mu + 4 * state.sigma, sampleMax),
  ]
  const curveX = range(0, 96).map((i) => domain[0] + (i * (domain[1] - domain[0])) / 95)
  const trueDensity = curveX.map(
    (x) => [x, normalPdf(x, state.mu, state.sigma)] as [number, number],
  )
  const fittedDensity = curveX.map(
    (x) => [x, normalPdf(x, sampleMean, sampleStd)] as [number, number],
  )

  renderDensityHistogram({
    ariaLabel: "Gaussian sample histogram with true and empirical density curves",
    binCount: state.binCount,
    curves: [
      {
        colorVar: "--chart-true",
        fallbackColor: "#9d3b42",
        points: trueDensity,
      },
      {
        colorVar: "--chart-fit",
        dashArray: "4 3",
        fallbackColor: "#2c7d54",
        points: fittedDensity,
        strokeWidth: 1.9,
      },
    ],
    domain,
    figure,
    samples: state.samples,
    statsText: `n=${state.n}  mean=${sampleMean.toFixed(2)}  std=${sampleStd.toFixed(2)}`,
  })
}

export const gaussianSampleFigure: InteractiveFigureDefinition<GaussianState> = {
  bindControls: ({ addCleanup, figure, render, state }) => {
    const slider = figure.querySelector<HTMLInputElement>(".gaussian-histogram-slider input")
    const updateSampleSize = () => {
      if (!slider) return

      state.n = sliderValueToSampleSize(Number(slider.value))
      regenerateGaussianSamples(state)
      render()
    }

    slider?.addEventListener("input", updateSampleSize)
    addCleanup(() => slider?.removeEventListener("input", updateSampleSize))
  },
  classNames: ["gaussian-histogram-figure"],
  cloneState: cloneGaussianState,
  readState: readGaussianState,
  render: renderGaussian,
  resample: (state) => {
    state.seed += 1
    regenerateGaussianSamples(state)
  },
  serializeState: (state) => ({
    bins: state.binCount,
    mu: state.mu,
    n: state.n,
    seed: state.seed,
    sigma: state.sigma,
  }),
  syncControls: syncGaussianControls,
  template: gaussianTemplate,
  type: "gaussian-sample-histogram",
}

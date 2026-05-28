import { range } from "d3"
import { InteractiveFigureDefinition } from "./core"
import { renderDensityHistogram } from "./densityHistogram"
import {
  clampSampleSize,
  logNormalPdf,
  makeLogNormalSamples,
  mean,
  readNumber,
  sampleSizeToSliderValue,
  sampleSliderMax,
  sliderValueToSampleSize,
  sortedQuantile,
  standardDeviation,
} from "./math"
import { sampleTicksHtml, toolbarHtml, tooltipHtml } from "./ui"

const logNormalSigmaSliderMax = 100
const logNormalMuSliderMax = 100

type LogNormalState = {
  binCount: number
  mu: number
  n: number
  seed: number
  sigma: number
  samples: number[]
}

function clampLogNormalMu(value: number) {
  return Math.min(2, Math.max(-1, value))
}

function clampLogNormalSigma(value: number) {
  return Math.min(1.6, Math.max(0.2, value))
}

function logNormalMuToSliderValue(value: number) {
  return Math.round(((clampLogNormalMu(value) + 1) / 3) * logNormalMuSliderMax)
}

function sliderValueToLogNormalMu(value: number) {
  const ratio = Math.min(1, Math.max(0, value / logNormalMuSliderMax))
  return clampLogNormalMu(-1 + ratio * 3)
}

function logNormalSigmaToSliderValue(value: number) {
  return Math.round(((clampLogNormalSigma(value) - 0.2) / 1.4) * logNormalSigmaSliderMax)
}

function sliderValueToLogNormalSigma(value: number) {
  const ratio = Math.min(1, Math.max(0, value / logNormalSigmaSliderMax))
  return clampLogNormalSigma(0.2 + ratio * 1.4)
}

function readLogNormalState(figure: HTMLElement): LogNormalState {
  const n = clampSampleSize(readNumber(figure.dataset.n, 1000))
  const state = {
    binCount: Math.round(readNumber(figure.dataset.bins, 30)),
    mu: clampLogNormalMu(readNumber(figure.dataset.mu, 0)),
    n,
    seed: Math.round(readNumber(figure.dataset.seed, 87)),
    sigma: clampLogNormalSigma(readNumber(figure.dataset.sigma, 0.5)),
    samples: [] as number[],
  }

  state.samples = makeLogNormalSamples(state.mu, state.sigma, state.n, state.seed)
  return state
}

function cloneLogNormalState(state: LogNormalState): LogNormalState {
  return {
    ...state,
    samples: [...state.samples],
  }
}

function regenerateLogNormalSamples(state: LogNormalState) {
  state.samples = makeLogNormalSamples(state.mu, state.sigma, state.n, state.seed)
}

function syncLogNormalControls(figure: HTMLElement, state: LogNormalState) {
  const nSlider = figure.querySelector<HTMLInputElement>(".lognormal-n-slider")
  const muSlider = figure.querySelector<HTMLInputElement>(".lognormal-mu-slider")
  const sigmaSlider = figure.querySelector<HTMLInputElement>(".lognormal-sigma-slider")
  const nOutput = figure.querySelector<HTMLOutputElement>(".lognormal-n")
  const muOutput = figure.querySelector<HTMLOutputElement>(".lognormal-mu")
  const sigmaOutput = figure.querySelector<HTMLOutputElement>(".lognormal-sigma")

  if (nSlider) nSlider.value = String(sampleSizeToSliderValue(state.n))
  if (muSlider) muSlider.value = String(logNormalMuToSliderValue(state.mu))
  if (sigmaSlider) sigmaSlider.value = String(logNormalSigmaToSliderValue(state.sigma))
  if (nOutput) nOutput.value = String(state.n)
  if (muOutput) muOutput.value = state.mu.toFixed(2)
  if (sigmaOutput) sigmaOutput.value = state.sigma.toFixed(2)

  figure.dataset.n = String(state.n)
  figure.dataset.mu = String(state.mu)
  figure.dataset.sigma = String(state.sigma)
  figure.dataset.seed = String(state.seed)
}

function logNormalTemplate({
  canExpand,
  id,
  state,
}: {
  canExpand: boolean
  id: string
  state: LogNormalState
}) {
  return `
    ${toolbarHtml("Log-normal samples", canExpand)}
    <div class="gaussian-histogram-plot"></div>
    <div class="gaussian-histogram-legend" aria-hidden="true">
      <span><i class="is-samples"></i>Samples</span>
      <span><i></i>GT density</span>
    </div>
    <div class="gaussian-histogram-stats"></div>
    <label class="gaussian-histogram-slider">
      <div class="gaussian-histogram-slider-row">
        <span>n</span>
        <input class="lognormal-n-slider" type="range" min="0" max="${sampleSliderMax}" step="1" value="${sampleSizeToSliderValue(
          state.n,
        )}" aria-describedby="${id}-n-ticks" />
        <output class="lognormal-n">${state.n}</output>
      </div>
      ${sampleTicksHtml(`${id}-n-ticks`)}
    </label>
    <label class="gaussian-histogram-slider lognormal-parameter-slider">
      <div class="gaussian-histogram-slider-row">
        <span>μ</span>
        <input class="lognormal-mu-slider" type="range" min="0" max="${logNormalMuSliderMax}" step="1" value="${logNormalMuToSliderValue(
          state.mu,
        )}" />
        <output class="lognormal-mu">${state.mu.toFixed(2)}</output>
      </div>
    </label>
    <label class="gaussian-histogram-slider lognormal-parameter-slider">
      <div class="gaussian-histogram-slider-row">
        <span>σ</span>
        <input class="lognormal-sigma-slider" type="range" min="0" max="${logNormalSigmaSliderMax}" step="1" value="${logNormalSigmaToSliderValue(
          state.sigma,
        )}" />
        <output class="lognormal-sigma">${state.sigma.toFixed(2)}</output>
      </div>
    </label>
    ${tooltipHtml()}
  `
}

function renderLogNormal(figure: HTMLElement, state: LogNormalState) {
  const sampleMean = mean(state.samples)
  const sampleStd = standardDeviation(state.samples, sampleMean)
  const sampleMax = sortedQuantile(state.samples, 0.995)
  const theoreticalMax = Math.exp(state.mu + 3.4 * state.sigma)
  const domain: [number, number] = [0, Math.max(1, sampleMax, theoreticalMax)]
  const densityDomainMax = domain[1]
  const curveX = range(0, 128).map((i) => {
    const min = Math.max(densityDomainMax / 1000, 0.001)
    return i === 0 ? 0 : min * (densityDomainMax / min) ** (i / 127)
  })
  const trueDensity = curveX.map(
    (x) => [x, logNormalPdf(x, state.mu, state.sigma)] as [number, number],
  )

  renderDensityHistogram({
    ariaLabel: "Log-normal samples with ground-truth density curve",
    binCount: state.binCount,
    curves: [
      {
        colorVar: "--chart-true",
        fallbackColor: "#9d3b42",
        points: trueDensity,
      },
    ],
    domain,
    figure,
    histogramSamples: state.samples.filter((sample) => sample <= domain[1]),
    samples: state.samples,
    statsText: `n=${state.n}  mean=${sampleMean.toFixed(2)}  std=${sampleStd.toFixed(2)}`,
  })
}

export const logNormalDensityFigure: InteractiveFigureDefinition<LogNormalState> = {
  bindControls: ({ addCleanup, figure, render, state }) => {
    const nSlider = figure.querySelector<HTMLInputElement>(".lognormal-n-slider")
    const muSlider = figure.querySelector<HTMLInputElement>(".lognormal-mu-slider")
    const sigmaSlider = figure.querySelector<HTMLInputElement>(".lognormal-sigma-slider")
    const updateSampleSize = () => {
      if (!nSlider) return

      state.n = sliderValueToSampleSize(Number(nSlider.value))
      regenerateLogNormalSamples(state)
      render()
    }
    const updateMu = () => {
      if (!muSlider) return

      state.mu = sliderValueToLogNormalMu(Number(muSlider.value))
      regenerateLogNormalSamples(state)
      render()
    }
    const updateSigma = () => {
      if (!sigmaSlider) return

      state.sigma = sliderValueToLogNormalSigma(Number(sigmaSlider.value))
      regenerateLogNormalSamples(state)
      render()
    }

    nSlider?.addEventListener("input", updateSampleSize)
    muSlider?.addEventListener("input", updateMu)
    sigmaSlider?.addEventListener("input", updateSigma)
    addCleanup(() => {
      nSlider?.removeEventListener("input", updateSampleSize)
      muSlider?.removeEventListener("input", updateMu)
      sigmaSlider?.removeEventListener("input", updateSigma)
    })
  },
  classNames: ["gaussian-histogram-figure", "lognormal-density-figure"],
  cloneState: cloneLogNormalState,
  readState: readLogNormalState,
  render: renderLogNormal,
  resample: (state) => {
    state.seed += 1
    regenerateLogNormalSamples(state)
  },
  serializeState: (state) => ({
    bins: state.binCount,
    mu: state.mu,
    n: state.n,
    seed: state.seed,
    sigma: state.sigma,
  }),
  syncControls: syncLogNormalControls,
  template: logNormalTemplate,
  type: "lognormal-density",
}

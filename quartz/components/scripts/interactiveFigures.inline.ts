import { registerInteractiveFigure, setupInteractiveFigures } from "./interactive-figures/core"
import { gaussianSampleFigure } from "./interactive-figures/gaussianSample"
import { logNormalDensityFigure } from "./interactive-figures/logNormalDensity"

registerInteractiveFigure(gaussianSampleFigure)
registerInteractiveFigure(logNormalDensityFigure)

document.addEventListener("nav", setupInteractiveFigures)

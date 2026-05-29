import { registerInteractiveFigure, setupInteractiveFigures } from "./interactive-figures/core"
import { boxMullerTransformFigure } from "./interactive-figures/boxMullerTransform"
import { gaussianSampleFigure } from "./interactive-figures/gaussianSample"
import { logNormalDensityFigure } from "./interactive-figures/logNormalDensity"

registerInteractiveFigure(boxMullerTransformFigure)
registerInteractiveFigure(gaussianSampleFigure)
registerInteractiveFigure(logNormalDensityFigure)

document.addEventListener("nav", setupInteractiveFigures)

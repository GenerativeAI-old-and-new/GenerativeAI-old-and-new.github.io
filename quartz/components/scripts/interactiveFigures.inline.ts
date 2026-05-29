import { registerInteractiveFigure, setupInteractiveFigures } from "./interactive-figures/core"
import { changeOfVariablesIntuitionFigure } from "./interactive-figures/changeOfVariablesIntuition"
import { gaussianSampleFigure } from "./interactive-figures/gaussianSample"
import { logNormalDensityFigure } from "./interactive-figures/logNormalDensity"

registerInteractiveFigure(changeOfVariablesIntuitionFigure)
registerInteractiveFigure(gaussianSampleFigure)
registerInteractiveFigure(logNormalDensityFigure)

document.addEventListener("nav", setupInteractiveFigures)

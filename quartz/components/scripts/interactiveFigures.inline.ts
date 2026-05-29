import { registerInteractiveFigure, setupInteractiveFigures } from "./interactive-figures/core"
import { changeOfVariablesIntuitionFigure } from "./interactive-figures/changeOfVariablesIntuition"
import { distributionLearningDiagramFigure } from "./interactive-figures/distributionLearningDiagram"
import { gaussianSampleFigure } from "./interactive-figures/gaussianSample"
import { logNormalDensityFigure } from "./interactive-figures/logNormalDensity"

registerInteractiveFigure(changeOfVariablesIntuitionFigure)
registerInteractiveFigure(distributionLearningDiagramFigure)
registerInteractiveFigure(gaussianSampleFigure)
registerInteractiveFigure(logNormalDensityFigure)

document.addEventListener("nav", setupInteractiveFigures)

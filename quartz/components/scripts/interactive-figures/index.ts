import { registerInteractiveFigure } from "./core"
import type { AnyFigureDefinition } from "./core"
import { changeOfVariablesIntuitionFigure } from "./changeOfVariablesIntuition"
import { distributionLearningDiagramFigure } from "./distributionLearningDiagram"
import { gaussianSampleFigure } from "./gaussianSample"
import { inverseCdfTransformFigure } from "./inverseCdfTransform"
import { logNormalDensityFigure } from "./logNormalDensity"
import { optimizerTrajectoryFigure } from "./optimizerTrajectory"

const builtInInteractiveFigures: AnyFigureDefinition[] = [
  changeOfVariablesIntuitionFigure,
  distributionLearningDiagramFigure,
  gaussianSampleFigure,
  inverseCdfTransformFigure,
  logNormalDensityFigure,
  optimizerTrajectoryFigure,
]

export function registerBuiltInInteractiveFigures() {
  for (const figure of builtInInteractiveFigures) {
    registerInteractiveFigure(figure)
  }
}

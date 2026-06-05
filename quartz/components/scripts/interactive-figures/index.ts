import { registerInteractiveFigure } from "./core"
import type { AnyFigureDefinition } from "./core"
import { changeOfVariablesIntuitionFigure } from "./changeOfVariablesIntuition"
import { distributionLearningDiagramFigure } from "./distributionLearningDiagram"
import { gaussianSampleFigure } from "./gaussianSample"
import { inverseCdfTransformFigure } from "./inverseCdfTransform"
import { logNormalDensityFigure } from "./logNormalDensity"
import { optimizerTrajectoryFigure } from "./optimizerTrajectory"
import { universalApproximationFigure } from "./universalApproximation"

const builtInInteractiveFigures: AnyFigureDefinition[] = [
  changeOfVariablesIntuitionFigure,
  distributionLearningDiagramFigure,
  gaussianSampleFigure,
  inverseCdfTransformFigure,
  logNormalDensityFigure,
  optimizerTrajectoryFigure,
  universalApproximationFigure,
]

export function registerBuiltInInteractiveFigures() {
  for (const figure of builtInInteractiveFigures) {
    registerInteractiveFigure(figure)
  }
}

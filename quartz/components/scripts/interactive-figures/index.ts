import { registerInteractiveFigure } from "./core"
import type { AnyFigureDefinition } from "./core"
import { changeOfVariablesIntuitionFigure } from "./changeOfVariablesIntuition"
import { distributionLearningDiagramFigure } from "./distributionLearningDiagram"
import { gaussianSampleFigure } from "./gaussianSample"
import { logNormalDensityFigure } from "./logNormalDensity"

const builtInInteractiveFigures: AnyFigureDefinition[] = [
  changeOfVariablesIntuitionFigure,
  distributionLearningDiagramFigure,
  gaussianSampleFigure,
  logNormalDensityFigure,
]

export function registerBuiltInInteractiveFigures() {
  for (const figure of builtInInteractiveFigures) {
    registerInteractiveFigure(figure)
  }
}

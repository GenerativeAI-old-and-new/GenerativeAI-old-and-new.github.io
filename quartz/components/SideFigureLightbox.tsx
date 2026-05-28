// @ts-ignore
import script from "./scripts/sidefigure.inline"
import { QuartzComponent, QuartzComponentConstructor } from "./types"

const SideFigureLightbox: QuartzComponent = () => null

SideFigureLightbox.afterDOMLoaded = script

export default (() => SideFigureLightbox) satisfies QuartzComponentConstructor

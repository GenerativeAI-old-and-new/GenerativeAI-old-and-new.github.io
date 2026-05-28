// @ts-ignore
import script from "./scripts/interactiveFigures.inline"
import { QuartzComponent, QuartzComponentConstructor } from "./types"

const InteractiveFigures: QuartzComponent = () => null

InteractiveFigures.afterDOMLoaded = script

export default (() => InteractiveFigures) satisfies QuartzComponentConstructor

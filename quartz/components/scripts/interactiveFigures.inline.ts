import { setupInteractiveFigures } from "./interactive-figures/core"
import { registerBuiltInInteractiveFigures } from "./interactive-figures"

registerBuiltInInteractiveFigures()

document.addEventListener("nav", setupInteractiveFigures)

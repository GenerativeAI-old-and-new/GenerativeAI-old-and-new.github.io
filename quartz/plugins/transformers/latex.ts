import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"
import rehypeMathjax from "rehype-mathjax/svg"
import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
//@ts-ignore
import rehypeTypst from "@myriaddreamin/rehype-typst"
import { QuartzTransformerPlugin } from "../types"
import { KatexOptions } from "katex"
import { Options as MathjaxOptions } from "rehype-mathjax/svg"
//@ts-ignore
import { Options as TypstOptions } from "@myriaddreamin/rehype-typst"

const require = createRequire(import.meta.url)
const katexCssPath = require.resolve("katex/dist/katex.min.css")
const katexFontCdn = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/fonts/"

function localKatexStylesheet() {
  return readFileSync(katexCssPath, "utf8").replaceAll("url(fonts/", `url(${katexFontCdn}`)
}

interface Options {
  renderEngine: "katex" | "mathjax" | "typst"
  customMacros: MacroType
  katexOptions: Omit<KatexOptions, "macros" | "output">
  mathJaxOptions: Omit<MathjaxOptions, "macros">
  typstOptions: TypstOptions
}

// mathjax macros
export type Args = boolean | number | string | null
interface MacroType {
  [key: string]: string | Args[]
}

export const Latex: QuartzTransformerPlugin<Partial<Options>> = (opts) => {
  const engine = opts?.renderEngine ?? "katex"
  const macros = opts?.customMacros ?? {}
  return {
    name: "Latex",
    markdownPlugins() {
      return [remarkMath]
    },
    htmlPlugins() {
      switch (engine) {
        case "katex": {
          return [[rehypeKatex, { output: "html", macros, ...(opts?.katexOptions ?? {}) }]]
        }
        case "typst": {
          return [[rehypeTypst, opts?.typstOptions ?? {}]]
        }
        default:
        case "mathjax": {
          return [
            [
              rehypeMathjax,
              {
                ...(opts?.mathJaxOptions ?? {}),
                tex: {
                  ...(opts?.mathJaxOptions?.tex ?? {}),
                  macros,
                },
              },
            ],
          ]
        }
      }
    },
    externalResources() {
      switch (engine) {
        case "katex":
          return {
            css: [{ content: localKatexStylesheet(), inline: true }],
            js: [
              {
                // fix copy behaviour: https://github.com/KaTeX/KaTeX/blob/main/contrib/copy-tex/README.md
                src: "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/copy-tex.min.js",
                loadTime: "afterDOMReady",
                contentType: "external",
              },
            ],
          }
      }
    },
  }
}

import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"
import rehypeMathjax from "rehype-mathjax/svg"
import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import { Root, Paragraph, PhrasingContent, RootContent, Parent, Text } from "mdast"
import { BuildVisitor, visit } from "unist-util-visit"
import { VFile } from "vfile"
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

type InlineMath = PhrasingContent & {
  type: "inlineMath"
  value: string
  position?: {
    start?: { offset?: number }
    end?: { offset?: number }
  }
}

function isText(node: PhrasingContent | undefined): node is Text {
  return node?.type === "text"
}

function trimParagraphChildren(children: PhrasingContent[]): PhrasingContent[] {
  const trimmed = [...children]

  while (isText(trimmed[0]) && trimmed[0].value.trim() === "") {
    trimmed.shift()
  }

  while (true) {
    const last = trimmed[trimmed.length - 1]
    if (!isText(last) || last.value.trim() !== "") break
    trimmed.pop()
  }

  if (isText(trimmed[0])) {
    const text = trimmed[0]
    trimmed[0] = { ...text, value: text.value.trimStart() }
  }

  const last = trimmed[trimmed.length - 1]
  if (isText(last)) {
    trimmed[trimmed.length - 1] = { ...last, value: last.value.trimEnd() }
  }

  return trimmed
}

function isDoubleDollarInlineMath(node: PhrasingContent, source: string): node is InlineMath {
  const maybeMath = node as Partial<InlineMath>
  if (maybeMath.type !== "inlineMath" || typeof maybeMath.value !== "string") return false

  const start = maybeMath.position?.start?.offset
  const end = maybeMath.position?.end?.offset
  if (start === undefined || end === undefined) return false

  const raw = source.slice(start, end)
  return raw.trimStart().startsWith("$$") && raw.trimEnd().endsWith("$$")
}

function createDisplayMath(value: string, position: InlineMath["position"]): RootContent {
  return {
    type: "math",
    meta: null,
    value,
    data: {
      hName: "pre",
      hChildren: [
        {
          type: "element",
          tagName: "code",
          properties: { className: ["language-math", "math-display"] },
          children: [{ type: "text", value }],
        },
      ],
    },
    position,
  } as RootContent
}

function mathjaxMacroPreamble(macros: MacroType): string {
  return Object.entries(macros)
    .map(([name, replacement]) => {
      if (Array.isArray(replacement)) {
        const [body, argCount] = replacement
        if (typeof body !== "string" || typeof argCount !== "number") return ""

        const args = Array.from({ length: argCount }, (_, i) => `#${i + 1}`).join("")
        return `\\def${name}${args}{${body}}`
      }

      return `\\def${name}{${replacement}}`
    })
    .filter(Boolean)
    .join("\n")
}

function injectMathjaxMacros(macros: MacroType) {
  const preamble = mathjaxMacroPreamble(macros)

  return () => {
    return (tree: Root) => {
      if (preamble === "") return

      visit(tree, ["math", "inlineMath"], ((node: { value?: string }) => {
        if (typeof node.value !== "string" || node.value.startsWith(preamble)) return
        node.value = `${preamble}\n${node.value}`
      }) as BuildVisitor<Root>)
    }
  }
}

function injectMathjaxMacrosHtml(macros: MacroType) {
  const preamble = mathjaxMacroPreamble(macros)

  return () => {
    return (tree: Root) => {
      if (preamble === "") return

      visit(tree, "element", (node: any) => {
        const className = node.properties?.className
        const classes = Array.isArray(className) ? className : []
        const isMathCode = classes.some((name) =>
          ["language-math", "math-display", "math-inline"].includes(String(name)),
        )

        if (!isMathCode || !Array.isArray(node.children)) return

        for (const child of node.children) {
          if (child.type !== "text" || typeof child.value !== "string") continue
          if (!child.value.startsWith(preamble)) child.value = `${preamble}\n${child.value}`
        }
      })
    }
  }
}

function splitQuotePrefix(line: string): [prefix: string, body: string] {
  const match = line.match(/^(\s*(?:>\s*)+)/)
  const prefix = match?.[0] ?? ""
  return [prefix, line.slice(prefix.length)]
}

function isFence(line: string): boolean {
  return /^ {0,3}(```|~~~)/.test(line)
}

function normalizeDoubleDollarBlocks(src: string): string {
  const output: string[] = []
  let inMath = false
  let mathPrefix = ""
  let inFenceBlock = false

  const pushBoundary = (prefix: string) => {
    const boundary = prefix.trim() === "" ? "" : prefix.trimEnd()
    if (output[output.length - 1] !== boundary) output.push(boundary)
  }

  for (const line of src.split(/\r?\n/)) {
    if (isFence(line)) {
      inFenceBlock = !inFenceBlock
      output.push(line)
      continue
    }

    if (inFenceBlock) {
      output.push(line)
      continue
    }

    const [linePrefix, body] = splitQuotePrefix(line)
    let cursor = 0
    let consumedMarker = false

    while (cursor <= body.length) {
      const marker = body.indexOf("$$", cursor)

      if (marker === -1) {
        const rest = body.slice(cursor)
        if (inMath) {
          output.push(mathPrefix + rest)
        } else if (consumedMarker) {
          if (rest.trim() !== "") output.push(linePrefix + rest.trimStart())
        } else {
          output.push(line)
        }
        break
      }

      const segment = body.slice(cursor, marker)
      consumedMarker = true

      if (inMath) {
        if (segment.trim() !== "") output.push(mathPrefix + segment.trimEnd())
        output.push(mathPrefix + "$$")
        pushBoundary(mathPrefix)
        inMath = false
        cursor = marker + 2
      } else {
        if (segment.trim() !== "") output.push(linePrefix + segment.trimEnd())
        pushBoundary(linePrefix)
        output.push(linePrefix + "$$")
        inMath = true
        mathPrefix = linePrefix
        cursor = marker + 2
      }
    }
  }

  return output.join("\n")
}

function normalizeAnnotationMacros(src: string): string {
  return src
    .replace(/\\ant\{\$([^$]+)\$\}/g, "\\ant{$1}")
    .replace(/\\ant\{([^{}\n$]+?),\s*\$([^$]+)\$\}/g, (_match, text, math) => {
      return `\\ant{\\text{${String(text).trim()}},\\,${math}}`
    })
}

function nodeSource(node: PhrasingContent, source: string): string {
  const start = node.position?.start.offset
  const end = node.position?.end.offset
  if (start !== undefined && end !== undefined) {
    return source.slice(start, end)
  }

  const maybeValue = node as Partial<InlineMath>
  return typeof maybeValue.value === "string" ? maybeValue.value : ""
}

function splitRawDoubleDollarMath() {
  return (tree: Root, file: VFile) => {
    const source = String(file.value ?? "")

    visit(tree, "paragraph", ((node: Paragraph, index: number, parent: Parent | null) => {
      if (!parent || index === undefined) return
      if (!node.children.some((child) => isText(child) && child.value.includes("$$"))) return

      const replacement: RootContent[] = []
      let paragraphChildren: PhrasingContent[] = []
      let mathValue = ""
      let inMath = false

      const flushParagraph = () => {
        const children = trimParagraphChildren(paragraphChildren)
        if (children.length > 0) {
          replacement.push({ type: "paragraph", children })
        }
        paragraphChildren = []
      }

      const flushMath = () => {
        const value = mathValue.trim()
        if (value !== "") {
          replacement.push(createDisplayMath(value, undefined))
        }
        mathValue = ""
      }

      const pushText = (text: Text, value: string) => {
        if (value === "") return
        if (inMath) {
          mathValue += value
        } else {
          paragraphChildren.push({ ...text, value, position: undefined })
        }
      }

      for (const child of node.children) {
        if (!isText(child)) {
          if (inMath) {
            mathValue += nodeSource(child, source)
          } else {
            paragraphChildren.push(child)
          }
          continue
        }

        let cursor = 0
        while (cursor <= child.value.length) {
          const marker = child.value.indexOf("$$", cursor)
          if (marker === -1) {
            pushText(child, child.value.slice(cursor))
            break
          }

          pushText(child, child.value.slice(cursor, marker))
          if (inMath) {
            flushMath()
            inMath = false
          } else {
            flushParagraph()
            inMath = true
            mathValue = ""
          }

          cursor = marker + 2
        }
      }

      if (inMath) return

      flushParagraph()
      parent.children.splice(index, 1, ...replacement)
    }) as BuildVisitor<Root, "paragraph">)
  }
}

function promoteDoubleDollarInlineMath() {
  return (tree: Root, file: VFile) => {
    const source = String(file.value ?? "")

    visit(tree, "paragraph", ((node: Paragraph, index: number, parent: Parent | null) => {
      if (!parent || index === undefined) return
      if (!node.children.some((child) => isDoubleDollarInlineMath(child, source))) return

      const replacement: RootContent[] = []
      let paragraphChildren: PhrasingContent[] = []

      const flushParagraph = () => {
        const children = trimParagraphChildren(paragraphChildren)
        if (children.length > 0) {
          replacement.push({ type: "paragraph", children })
        }
        paragraphChildren = []
      }

      for (const child of node.children) {
        if (isDoubleDollarInlineMath(child, source)) {
          flushParagraph()
          replacement.push(createDisplayMath(child.value, child.position))
        } else {
          paragraphChildren.push(child)
        }
      }

      flushParagraph()
      parent.children.splice(index, 1, ...replacement)
    }) as BuildVisitor<Root, "paragraph">)
  }
}

export const Latex: QuartzTransformerPlugin<Partial<Options>> = (opts) => {
  const engine = opts?.renderEngine ?? "katex"
  const macros = opts?.customMacros ?? {}
  return {
    name: "Latex",
    textTransform(_ctx, src) {
      return normalizeAnnotationMacros(normalizeDoubleDollarBlocks(src))
    },
    markdownPlugins() {
      return [
        remarkMath,
        splitRawDoubleDollarMath,
        promoteDoubleDollarInlineMath,
        ...(engine === "mathjax" ? [injectMathjaxMacros(macros)] : []),
      ]
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
            injectMathjaxMacrosHtml(macros),
            [
              rehypeMathjax,
              {
                ...(opts?.mathJaxOptions ?? {}),
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

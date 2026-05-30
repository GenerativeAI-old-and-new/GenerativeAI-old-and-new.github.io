import { Element, ElementContent, Root, RootContent, Text } from "hast"
import { toString } from "hast-util-to-string"
import { VFile } from "vfile"
import { styleText } from "util"
import { QuartzTransformerPlugin } from "../types"
import { ProcessedContent } from "../vfile"
import { BuildCtx } from "../../util/ctx"
import { escapeHTML } from "../../util/escape"
import { FullSlug, SimpleSlug, RelativeURL, resolveRelative, simplifySlug } from "../../util/path"

const numberedKinds = new Set([
  "definition",
  "remark",
  "note",
  "background",
  "notation",
  "example",
  "problem",
  "theorem",
  "lemma",
  "proposition",
  "corollary",
])

const displayKind: Record<string, string> = {
  definition: "Definition",
  remark: "Remark",
  note: "Note",
  background: "Background",
  notation: "Notation",
  example: "Example",
  problem: "Problem",
  theorem: "Theorem",
  lemma: "Lemma",
  proposition: "Proposition",
  corollary: "Corollary",
}

const ignoredRefParents = new Set(["a", "code", "pre", "script", "style"])
const generatedTitleClasses = new Set([
  "callout-number",
  "callout-title-label",
  "callout-title-punctuation",
  "callout-title-inner",
])
const labelPattern = "[A-Za-z][A-Za-z0-9:_-]*(?:\\.[A-Za-z0-9:_-]+)*"
const trailingLabelRegex = new RegExp(`\\s*\\{#(${labelPattern})\\}\\s*$`)
const refRegex = new RegExp(`(^|[^\\w@])@(${labelPattern})`, "g")

type ParentNode = Root | Element

type TheoremEntry = {
  id: string
  kind: string
  label: string
  number: string
  refText: string
  slug: FullSlug
}

type TheoremDefinition = {
  callout: Element
  kind: string
  label?: string
  title: string
}

function classList(node: Element): string[] {
  const raw = node.properties?.className ?? node.properties?.class
  if (Array.isArray(raw)) return raw.map(String)
  if (typeof raw === "string") return raw.split(/\s+/).filter(Boolean)
  return []
}

function hasClass(node: Element, className: string): boolean {
  return classList(node).includes(className)
}

function getStringProperty(node: Element, key: string): string {
  const value = node.properties?.[key] ?? node.properties?.[dataAttributeCamelName(key)]
  if (typeof value === "string") return value
  if (Array.isArray(value)) return value.map(String).join(" ")
  return ""
}

function dataAttributeCamelName(key: string): string {
  return key.replace(/-([a-z])/g, (_, char: string) => char.toUpperCase())
}

function findDirectChildByClass(node: Element, className: string): Element | undefined {
  return node.children.find(
    (child): child is Element => child.type === "element" && hasClass(child, className),
  )
}

function findLastTextNode(node: Element): Text | undefined {
  for (let i = node.children.length - 1; i >= 0; i--) {
    const child = node.children[i]
    if (child.type === "text") return child
    if (child.type === "element") {
      const found = findLastTextNode(child)
      if (found) return found
    }
  }
}

function removeTrailingLabel(node: Element): string | undefined {
  const text = findLastTextNode(node)
  if (!text) return

  const match = text.value.match(trailingLabelRegex)
  if (!match) return

  text.value = text.value.replace(trailingLabelRegex, "").trimEnd()
  return match[1]
}

function normalizeTitle(title: string, kind: string): string {
  const normalized = title.replace(/\s+/g, " ").trim()
  return normalized === displayKind[kind] ? "" : normalized
}

function textNode(value: string): Text {
  return { type: "text", value }
}

function spanNode(className: string, value: string): Element {
  return {
    type: "element",
    tagName: "span",
    properties: { className: [className] },
    children: [textNode(value)],
  }
}

function theoremRefPlaceholder(label: string): Element {
  return {
    type: "element",
    tagName: "span",
    properties: {
      className: ["theorem-ref", "pending"],
      "data-theorem-ref": label,
    },
    children: [textNode(`@${label}`)],
  }
}

function replaceRefsInText(value: string): Array<ElementContent | RootContent> | undefined {
  const replacements: Array<ElementContent | RootContent> = []
  let lastIndex = 0
  let matched = false

  for (const match of value.matchAll(refRegex)) {
    const prefix = match[1] ?? ""
    const label = match[2]
    const refStart = match.index! + prefix.length
    const refEnd = refStart + label.length + 1

    if (prefix.endsWith("[")) continue

    if (refStart > lastIndex) {
      replacements.push(textNode(value.slice(lastIndex, refStart)))
    }

    replacements.push(theoremRefPlaceholder(label))
    lastIndex = refEnd
    matched = true
  }

  if (!matched) return
  if (lastIndex < value.length) {
    replacements.push(textNode(value.slice(lastIndex)))
  }

  return replacements
}

function walkForRefs(parent: ParentNode, skip = false) {
  for (let i = 0; i < parent.children.length; i++) {
    const child = parent.children[i]
    if (child.type === "text") {
      if (skip) continue

      const replacements = replaceRefsInText(child.value)
      if (replacements) {
        parent.children.splice(i, 1, ...(replacements as ElementContent[]))
        i += replacements.length - 1
      }
    } else if (child.type === "element") {
      const isFigureRef = getStringProperty(child, "data-figure-ref") !== ""
      walkForRefs(child, skip || ignoredRefParents.has(child.tagName) || isFigureRef)
    }
  }
}

function walkElements(parent: ParentNode, callback: (node: Element) => void) {
  for (const child of parent.children) {
    if (child.type !== "element") continue
    callback(child)
    walkElements(child, callback)
  }
}

export function markTheoremReferences(tree: Root, _file?: VFile) {
  walkElements(tree, (node) => {
    const kind = getStringProperty(node, "data-callout")
    if (!numberedKinds.has(kind)) return

    const titleNode = findDirectChildByClass(node, "callout-title")
    const titleInner = titleNode
      ? findDirectChildByClass(titleNode, "callout-title-inner")
      : undefined
    const label = titleInner ? removeTrailingLabel(titleInner) : undefined
    const metadataTitle = titleNode ? getStringProperty(titleNode, "data-callout-metadata") : ""
    const inlineTitle = titleInner ? toString(titleInner) : ""
    const title = normalizeTitle(metadataTitle || inlineTitle, kind)

    node.properties = {
      ...node.properties,
      "data-theorem-kind": kind,
      ...(title ? { "data-theorem-title": title } : {}),
      ...(label ? { "data-theorem-label": label } : {}),
    }
  })

  walkForRefs(tree)
}

function modulePrefix(slug: FullSlug): string | undefined {
  const segment = slug.split("/").at(-1) ?? ""
  const match = segment.match(/^(\d+)/)
  if (!match) return
  return String(Number(match[1]))
}

function theoremNumber(slug: FullSlug, count: number): string {
  const prefix = modulePrefix(slug)
  return prefix ? `${prefix}.${count}` : String(count)
}

function safeId(value: string): string {
  return value
    .trim()
    .replace(/[^A-Za-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase()
}

function generatedId(kind: string, number: string): string {
  return `${kind}-${number.replace(/\./g, "-")}`
}

function collectDefinitions(root: Root): TheoremDefinition[] {
  const definitions: TheoremDefinition[] = []
  walkElements(root, (node) => {
    const kind = getStringProperty(node, "data-theorem-kind")
    if (!numberedKinds.has(kind)) return

    definitions.push({
      callout: node,
      kind,
      label: getStringProperty(node, "data-theorem-label") || undefined,
      title: getStringProperty(node, "data-theorem-title"),
    })
  })
  return definitions
}

function replaceTitle(title: Element, entry: TheoremEntry, titleText: string) {
  const rest = title.children.filter((child) => {
    if (child.type !== "element") return true
    return !classList(child).some((className) => generatedTitleClasses.has(className))
  })

  delete title.properties?.["data-callout-metadata"]
  delete title.properties?.dataCalloutMetadata
  title.children = [
    spanNode("callout-number", entry.refText),
    ...(titleText ? [spanNode("callout-title-label", `\u00a0(${titleText})`)] : []),
    spanNode("callout-title-punctuation", "."),
    ...rest,
  ]
}

function warningText(message: string): string {
  return styleText("yellow", `Warning: ${message}`)
}

function linkHref(currentSlug: FullSlug, target: TheoremEntry): RelativeURL | string {
  if (currentSlug === target.slug) return `#${target.id}`
  return `${resolveRelative(currentSlug, target.slug)}#${target.id}` as RelativeURL
}

function updateLinks(file: VFile, targetSlug: FullSlug) {
  if (file.data.slug === targetSlug) return

  const simpleSlug = simplifySlug(targetSlug) as SimpleSlug
  const links = new Set(file.data.links ?? [])
  links.add(simpleSlug)
  file.data.links = [...links]
}

function finalizeRefs(
  root: Root,
  file: VFile,
  index: Map<string, TheoremEntry>,
  warn: (message: string) => void,
) {
  const slug = file.data.slug!
  walkElements(root, (node) => {
    const label = getStringProperty(node, "data-theorem-ref")
    if (!label) return

    const target = index.get(label)
    if (!target) {
      warn(`Unresolved theorem reference @${label} in ${slug}`)
      node.tagName = "span"
      node.properties = {
        className: ["theorem-ref", "unresolved"],
        "data-theorem-ref": label,
      }
      node.children = [textNode(`@${label}`)]
      return
    }

    node.tagName = "a"
    node.properties = {
      href: linkHref(slug, target),
      className: ["theorem-ref", "internal"],
      "data-slug": target.slug,
      "data-theorem-ref": label,
    }
    node.children = [textNode(target.refText)]
    updateLinks(file, target.slug)
  })
}

function refreshSearchText(root: Root, file: VFile) {
  file.data.text = escapeHTML(toString(root))
}

export function finalizeTheoremReferences(
  ctx: Pick<BuildCtx, "allSlugs">,
  content: ProcessedContent[],
  warn: (message: string) => void = (message) => console.warn(warningText(message)),
): Set<FullSlug> {
  const index = new Map<string, TheoremEntry>()
  const affectedSlugs = new Set<FullSlug>()

  for (const [root, file] of content) {
    const slug = file.data.slug!
    const counts = new Map<string, number>()

    for (const definition of collectDefinitions(root)) {
      const count = (counts.get(definition.kind) ?? 0) + 1
      counts.set(definition.kind, count)

      const number = theoremNumber(slug, count)
      const refText = `${displayKind[definition.kind]} ${number}`
      const id = definition.label ? safeId(definition.label) : generatedId(definition.kind, number)
      const entry: TheoremEntry = {
        id,
        kind: definition.kind,
        label: definition.label ?? "",
        number,
        refText,
        slug,
      }

      definition.callout.properties = {
        ...definition.callout.properties,
        id,
        "data-theorem-number": number,
      }

      const title = findDirectChildByClass(definition.callout, "callout-title")
      if (title) {
        replaceTitle(title, entry, definition.title)
      }

      affectedSlugs.add(slug)
      if (!definition.label) continue

      const existing = index.get(definition.label)
      if (existing) {
        throw new Error(
          `Duplicate theorem label "${definition.label}" in ${slug}; first defined in ${existing.slug}`,
        )
      }
      index.set(definition.label, entry)
    }
  }

  for (const [root, file] of content) {
    const textBefore = toString(root)
    finalizeRefs(root, file, index, warn)
    if (toString(root) !== textBefore) {
      affectedSlugs.add(file.data.slug!)
    }
    refreshSearchText(root, file)
  }

  // Touch ctx so tests can pass a minimal context while keeping the production
  // signature tied to the build pipeline.
  void ctx.allSlugs
  return affectedSlugs
}

export const TheoremReferences: QuartzTransformerPlugin = () => {
  return {
    name: "TheoremReferences",
    htmlPlugins() {
      return [
        () => {
          return (tree: Root, file) => {
            markTheoremReferences(tree, file)
          }
        },
      ]
    },
  }
}

declare module "vfile" {
  interface DataMap {
    links: SimpleSlug[]
    text: string
  }
}

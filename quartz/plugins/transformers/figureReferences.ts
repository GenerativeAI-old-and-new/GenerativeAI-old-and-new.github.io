import { Element, ElementContent, Root, RootContent, Text } from "hast"
import { toString } from "hast-util-to-string"
import { VFile } from "vfile"
import { styleText } from "util"
import { QuartzTransformerPlugin } from "../types"
import { ProcessedContent } from "../vfile"
import { BuildCtx } from "../../util/ctx"
import { escapeHTML } from "../../util/escape"
import { FullSlug, RelativeURL, SimpleSlug, resolveRelative, simplifySlug } from "../../util/path"

const figureLabelPattern = "fig:[A-Za-z][A-Za-z0-9:_-]*(?:\\.[A-Za-z0-9:_-]+)*"
const trailingLabelRegex = new RegExp(`\\s*\\{#(${figureLabelPattern})\\}\\s*$`)
const refRegex = new RegExp(`(^|[^\\w@])@(${figureLabelPattern})`, "g")
const ignoredRefParents = new Set(["a", "code", "pre", "script", "style"])
const genericAltText = new Set(["image", "figure", "fig"])

type ParentNode = Root | Element

type FigureEntry = {
  id: string
  label: string
  number: string
  refText: string
  slug: FullSlug
}

type FigureDefinition = {
  figure: Element
  label?: string
  title: string
}

function dataAttributeCamelName(key: string): string {
  return key.replace(/-([a-z])/g, (_, char: string) => char.toUpperCase())
}

function getStringProperty(node: Element, key: string): string {
  const value = node.properties?.[key] ?? node.properties?.[dataAttributeCamelName(key)]
  if (typeof value === "string") return value
  if (Array.isArray(value)) return value.map(String).join(" ")
  return ""
}

function classList(node: Element): string[] {
  const raw = node.properties?.className ?? node.properties?.class
  if (Array.isArray(raw)) return raw.map(String)
  if (typeof raw === "string") return raw.split(/\s+/).filter(Boolean)
  return []
}

function addClass(node: Element, className: string) {
  const classes = classList(node)
  if (!classes.includes(className)) classes.push(className)
  node.properties = { ...node.properties, className: classes }
}

function hasClass(node: Element, className: string): boolean {
  return classList(node).includes(className)
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

function findDirectChildByTag(node: Element, tagName: string): Element | undefined {
  return node.children.find((child): child is Element => {
    return child.type === "element" && child.tagName === tagName
  })
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

function extractTrailingLabel(value: string): { label?: string; text: string } {
  const match = value.match(trailingLabelRegex)
  if (!match) return { text: value }

  return {
    label: match[1],
    text: value.replace(trailingLabelRegex, "").trimEnd(),
  }
}

function removeTrailingLabel(node: Element): string | undefined {
  const text = findLastTextNode(node)
  if (!text) return

  const { label, text: nextText } = extractTrailingLabel(text.value)
  if (!label) return

  text.value = nextText
  return label
}

function normalizeCaption(value: string): string {
  const caption = value.replace(/\s+/g, " ").trim()
  return genericAltText.has(caption.toLowerCase()) ? "" : caption
}

function isWhitespaceText(child: ElementContent | RootContent): child is Text {
  return child.type === "text" && child.value.trim() === ""
}

function isElement(child: ElementContent | RootContent, tagName: string): child is Element {
  return child.type === "element" && child.tagName === tagName
}

function shouldIgnoreFigure(node: Element): boolean {
  return getStringProperty(node, "data-rehype-pretty-code-figure") !== ""
}

function imageParagraphPayload(node: Element):
  | {
      caption?: Element
      image: Element
    }
  | undefined {
  if (node.tagName !== "p") return

  const children = node.children.filter((child) => !isWhitespaceText(child))
  if (children.length === 1 && isElement(children[0], "img")) {
    return { image: children[0] }
  }

  if (children.length === 2 && isElement(children[0], "img") && isElement(children[1], "em")) {
    return { image: children[0], caption: children[1] }
  }
}

function figureFromImageParagraph(node: Element): Element | undefined {
  const payload = imageParagraphPayload(node)
  if (!payload) return

  const rawAlt = getStringProperty(payload.image, "alt")
  const alt = extractTrailingLabel(rawAlt)
  const rawCaption = payload.caption ? extractTrailingLabel(toString(payload.caption)) : undefined
  const label = rawCaption?.label ?? alt.label
  const caption = normalizeCaption(rawCaption?.text ?? alt.text)

  payload.image.properties = {
    ...payload.image.properties,
    alt: caption,
  }

  const children: ElementContent[] = [payload.image]
  if (caption) {
    children.push({
      type: "element",
      tagName: "figcaption",
      properties: {},
      children: [textNode(caption)],
    })
  }

  return {
    type: "element",
    tagName: "figure",
    properties: {
      className: ["image-figure"],
      ...(caption ? { "data-figure-title": caption } : {}),
      ...(label ? { "data-figure-label": label } : {}),
    },
    children,
  }
}

function normalizeImageFigures(parent: ParentNode) {
  for (let i = 0; i < parent.children.length; i++) {
    const child = parent.children[i]
    if (child.type !== "element") continue

    const figure = figureFromImageParagraph(child)
    if (figure) {
      parent.children.splice(i, 1, figure as ElementContent)
      normalizeImageFigures(figure)
      continue
    }

    normalizeImageFigures(child)
  }
}

function markFigureMetadata(tree: Root) {
  walkElements(tree, (node) => {
    if (node.tagName !== "figure" || shouldIgnoreFigure(node)) return

    const caption = findDirectChildByTag(node, "figcaption")
    const label =
      getStringProperty(node, "data-figure-label") || (caption ? removeTrailingLabel(caption) : "")
    const captionText = caption ? normalizeCaption(toString(caption)) : ""
    const title = getStringProperty(node, "data-figure-title") || captionText

    if (!label && !title) return

    addClass(node, "numbered-figure")
    node.properties = {
      ...node.properties,
      ...(label ? { "data-figure-label": label } : {}),
      ...(title ? { "data-figure-title": title } : {}),
    }
  })
}

function figureRefPlaceholder(label: string): Element {
  return {
    type: "element",
    tagName: "span",
    properties: {
      className: ["figure-ref", "pending"],
      "data-figure-ref": label,
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

    replacements.push(figureRefPlaceholder(label))
    lastIndex = refEnd
    matched = true
  }

  if (!matched) return
  if (lastIndex < value.length) {
    replacements.push(textNode(value.slice(lastIndex)))
  }

  return replacements
}

function hasDataProperty(node: Element, key: string): boolean {
  return getStringProperty(node, key) !== ""
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
      walkForRefs(
        child,
        skip || ignoredRefParents.has(child.tagName) || hasDataProperty(child, "data-figure-ref"),
      )
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

export function markFigureReferences(tree: Root, _file?: VFile) {
  normalizeImageFigures(tree)
  markFigureMetadata(tree)
  walkForRefs(tree)
}

function modulePrefix(slug: FullSlug): string | undefined {
  const segment = slug.split("/").at(-1) ?? ""
  const match = segment.match(/^(\d+)/)
  if (!match) return
  return String(Number(match[1]))
}

function figureNumber(slug: FullSlug, count: number): string {
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

function generatedId(number: string): string {
  return `figure-${number.replace(/\./g, "-")}`
}

function collectDefinitions(root: Root): FigureDefinition[] {
  const definitions: FigureDefinition[] = []
  walkElements(root, (node) => {
    if (node.tagName !== "figure" || shouldIgnoreFigure(node)) return
    if (!hasClass(node, "numbered-figure")) return

    definitions.push({
      figure: node,
      label: getStringProperty(node, "data-figure-label") || undefined,
      title: getStringProperty(node, "data-figure-title"),
    })
  })
  return definitions
}

function replaceCaption(figure: Element, entry: FigureEntry, titleText: string) {
  let caption = findDirectChildByTag(figure, "figcaption")
  if (!caption) {
    caption = {
      type: "element",
      tagName: "figcaption",
      properties: {},
      children: [],
    }
    figure.children.push(caption)
  }

  caption.children = [
    spanNode("figure-number", entry.refText),
    spanNode("figure-title-punctuation", "."),
    ...(titleText ? [textNode(` ${titleText}`)] : []),
  ]
}

function warningText(message: string): string {
  return styleText("yellow", `Warning: ${message}`)
}

function linkHref(currentSlug: FullSlug, target: FigureEntry): RelativeURL | string {
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
  index: Map<string, FigureEntry>,
  warn: (message: string) => void,
) {
  const slug = file.data.slug!
  walkElements(root, (node) => {
    const label = getStringProperty(node, "data-figure-ref")
    if (!label) return

    const target = index.get(label)
    if (!target) {
      warn(`Unresolved figure reference @${label} in ${slug}`)
      node.tagName = "span"
      node.properties = {
        className: ["figure-ref", "unresolved"],
        "data-figure-ref": label,
      }
      node.children = [textNode(`@${label}`)]
      return
    }

    node.tagName = "a"
    node.properties = {
      href: linkHref(slug, target),
      className: ["figure-ref", "internal"],
      "data-slug": target.slug,
      "data-figure-ref": label,
    }
    node.children = [textNode(target.refText)]
    updateLinks(file, target.slug)
  })
}

function refreshSearchText(root: Root, file: VFile) {
  file.data.text = escapeHTML(toString(root))
}

export function finalizeFigureReferences(
  ctx: Pick<BuildCtx, "allSlugs">,
  content: ProcessedContent[],
  warn: (message: string) => void = (message) => console.warn(warningText(message)),
): Set<FullSlug> {
  const index = new Map<string, FigureEntry>()
  const affectedSlugs = new Set<FullSlug>()

  for (const [root, file] of content) {
    const slug = file.data.slug!
    let count = 0

    for (const definition of collectDefinitions(root)) {
      count += 1

      const number = figureNumber(slug, count)
      const refText = `Figure ${number}`
      const id = definition.label ? safeId(definition.label) : generatedId(number)
      const entry: FigureEntry = {
        id,
        label: definition.label ?? "",
        number,
        refText,
        slug,
      }

      definition.figure.properties = {
        ...definition.figure.properties,
        id,
        "data-figure-number": number,
      }
      replaceCaption(definition.figure, entry, definition.title)

      affectedSlugs.add(slug)
      if (!definition.label) continue

      const existing = index.get(definition.label)
      if (existing) {
        throw new Error(
          `Duplicate figure label "${definition.label}" in ${slug}; first defined in ${existing.slug}`,
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

  void ctx.allSlugs
  return affectedSlugs
}

export const FigureReferences: QuartzTransformerPlugin = () => {
  return {
    name: "FigureReferences",
    htmlPlugins() {
      return [
        () => {
          return (tree: Root, file) => {
            markFigureReferences(tree, file)
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

import assert from "node:assert/strict"
import test, { describe } from "node:test"
import { Element, Root } from "hast"
import { toString } from "hast-util-to-string"
import { VFile } from "vfile"
import { FilePath, FullSlug } from "../../util/path"
import { ProcessedContent } from "../vfile"
import { finalizeFigureReferences, markFigureReferences } from "./figureReferences"

function text(value: string) {
  return { type: "text" as const, value }
}

function element(
  tagName: string,
  properties: Element["properties"],
  children: Element["children"],
) {
  return { type: "element" as const, tagName, properties, children }
}

function imageParagraph(alt: string, src = "/assets/example.png"): Element {
  return element("p", {}, [element("img", { src, alt }, [])])
}

function paragraph(value: string): Element {
  return element("p", {}, [text(value)])
}

function rawFigure(caption: string): Element {
  return element("figure", {}, [
    element("img", { src: "/assets/example.png", alt: "" }, []),
    element("figcaption", {}, [text(caption)]),
  ])
}

function file(slug: string): VFile {
  const vfile = new VFile("")
  vfile.data.slug = slug as FullSlug
  vfile.data.relativePath = `${slug}.md` as FilePath
  vfile.data.links = []
  return vfile
}

function process(root: Root, slug = "modules/01-probability-basics"): ProcessedContent {
  const vfile = file(slug)
  markFigureReferences(root, vfile)
  finalizeFigureReferences({ allSlugs: [vfile.data.slug!] }, [[root, vfile]], () => {})
  return [root, vfile]
}

function firstElement(root: Root, predicate: (node: Element) => boolean): Element | undefined {
  const stack = [...root.children]
  while (stack.length > 0) {
    const node = stack.shift()
    if (node?.type !== "element") continue
    if (predicate(node)) return node
    stack.unshift(...node.children)
  }
}

describe("figure references", () => {
  test("numbers labelled markdown images and resolves internal references", () => {
    const root: Root = {
      type: "root",
      children: [
        imageParagraph("Posterior density {#fig:posterior}"),
        paragraph("See @fig:posterior."),
      ],
    }

    process(root)

    const figure = root.children[0] as Element
    const image = firstElement(root, (node) => node.tagName === "img")
    const link = firstElement(root, (node) => node.tagName === "a")

    assert.equal(figure.tagName, "figure")
    assert.equal(figure.properties?.id, "fig-posterior")
    assert.equal(figure.properties?.["data-figure-label"], "fig:posterior")
    assert.equal(figure.properties?.["data-figure-number"], "1.1")
    assert.equal(image?.properties?.alt, "Posterior density")
    assert.match(toString(figure), /Figure 1\.1\. Posterior density/)
    assert.equal(link?.properties?.href, "#fig-posterior")
    assert.deepEqual(link?.properties?.className, ["figure-ref", "internal"])
    assert.equal(toString(link!), "Figure 1.1")
  })

  test("does not number generic unlabelled image placeholders", () => {
    const root: Root = { type: "root", children: [imageParagraph("image")] }

    process(root)

    const figure = root.children[0] as Element
    const image = firstElement(root, (node) => node.tagName === "img")

    assert.equal(figure.tagName, "figure")
    assert.equal(figure.properties?.["data-figure-number"], undefined)
    assert.equal(
      firstElement(root, (node) => node.tagName === "figcaption"),
      undefined,
    )
    assert.equal(image?.properties?.alt, "")
  })

  test("numbers raw figures with labelled captions", () => {
    const root: Root = {
      type: "root",
      children: [rawFigure("Sampling paths {#fig:paths}")],
    }

    process(root, "notes/sampling")

    const figure = root.children[0] as Element
    assert.equal(figure.properties?.id, "fig-paths")
    assert.equal(figure.properties?.["data-figure-label"], "fig:paths")
    assert.match(toString(figure), /Figure 1\. Sampling paths/)
    assert.doesNotMatch(toString(figure), /\{#fig:paths\}/)
  })

  test("resolves cross-page references and records outgoing links", () => {
    const sourceRoot: Root = {
      type: "root",
      children: [imageParagraph("Posterior density {#fig:posterior}")],
    }
    const targetRoot: Root = { type: "root", children: [paragraph("See @fig:posterior.")] }
    const sourceFile = file("modules/01-probability-basics")
    const targetFile = file("modules/02-deep-learning-basics")
    const content: ProcessedContent[] = [
      [sourceRoot, sourceFile],
      [targetRoot, targetFile],
    ]

    markFigureReferences(sourceRoot, sourceFile)
    markFigureReferences(targetRoot, targetFile)
    finalizeFigureReferences(
      { allSlugs: [sourceFile.data.slug!, targetFile.data.slug!] },
      content,
      () => {},
    )

    const link = firstElement(targetRoot, (node) => node.tagName === "a")
    assert.equal(link?.properties?.href, "../modules/01-probability-basics#fig-posterior")
    assert.deepEqual(targetFile.data.links, ["modules/01-probability-basics"])
  })

  test("marks unresolved references and emits a warning", () => {
    const root: Root = { type: "root", children: [paragraph("See @fig:missing.")] }
    const vfile = file("modules/01-probability-basics")
    const warnings: string[] = []

    markFigureReferences(root, vfile)
    finalizeFigureReferences({ allSlugs: [vfile.data.slug!] }, [[root, vfile]], (msg) =>
      warnings.push(msg),
    )

    const unresolved = firstElement(root, (node) => node.tagName === "span")
    assert.deepEqual(unresolved?.properties?.className, ["figure-ref", "unresolved"])
    assert.equal(toString(unresolved!), "@fig:missing")
    assert.equal(warnings.length, 1)
  })

  test("throws on duplicate labels", () => {
    const root: Root = {
      type: "root",
      children: [imageParagraph("First {#fig:dup}"), imageParagraph("Second {#fig:dup}")],
    }
    const vfile = file("modules/01-probability-basics")

    markFigureReferences(root, vfile)

    assert.throws(
      () => finalizeFigureReferences({ allSlugs: [vfile.data.slug!] }, [[root, vfile]], () => {}),
      /Duplicate figure label "fig:dup"/,
    )
  })
})

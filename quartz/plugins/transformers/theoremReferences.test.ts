import assert from "node:assert/strict"
import test, { describe } from "node:test"
import { Element, Root } from "hast"
import { toString } from "hast-util-to-string"
import { VFile } from "vfile"
import { FullSlug, FilePath } from "../../util/path"
import { ProcessedContent } from "../vfile"
import { finalizeTheoremReferences, markTheoremReferences } from "./theoremReferences"

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

function callout(kind: string, metadata: string, titleInner: string): Element {
  return element(
    "blockquote",
    {
      className: ["callout", kind],
      "data-callout": kind,
      "data-callout-metadata": metadata,
    },
    [
      element("div", { className: ["callout-title"], "data-callout-metadata": metadata }, [
        element("div", { className: ["callout-icon"] }, []),
        element("div", { className: ["callout-title-inner"] }, [
          element("p", {}, [text(titleInner)]),
        ]),
      ]),
      element("div", { className: ["callout-content"] }, [element("p", {}, [text("Body.")])]),
    ],
  )
}

function paragraph(value: string): Element {
  return element("p", {}, [text(value)])
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
  markTheoremReferences(root, vfile)
  finalizeTheoremReferences({ allSlugs: [vfile.data.slug!] }, [[root, vfile]], () => {})
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

describe("theorem references", () => {
  test("parses callout labels and removes the raw label token", () => {
    const root: Root = {
      type: "root",
      children: [callout("theorem", "Change of Variables", "Theorem {#thm:change-vars}")],
    }

    process(root)

    const theorem = root.children[0] as Element
    const title = theorem.children[0] as Element
    assert.equal(theorem.properties?.id, "thm-change-vars")
    assert.equal(theorem.properties?.["data-theorem-label"], "thm:change-vars")
    assert.equal(title.properties?.["data-callout-metadata"], undefined)
    assert.match(toString(title), /Theorem 1\.1\u00a0\(Change of Variables\)\./)
    assert.match(toString(theorem), /Theorem 1\.1\s\(Change of Variables\)\./)
    assert.doesNotMatch(toString(theorem), /\{#thm:change-vars\}/)
  })

  test("recognizes normalized HAST data attribute property names", () => {
    const theorem = callout("theorem", "Change of Variables", "Theorem {#thm:change-vars}")
    theorem.properties = {
      className: ["callout", "theorem"],
      dataCallout: "theorem",
      dataCalloutMetadata: "Change of Variables",
    }
    ;(theorem.children[0] as Element).properties = {
      className: ["callout-title"],
      dataCalloutMetadata: "Change of Variables",
    }
    const root: Root = { type: "root", children: [theorem] }

    process(root)

    assert.equal(theorem.properties?.id, "thm-change-vars")
    assert.match(toString(theorem), /Theorem 1\.1\s\(Change of Variables\)\./)
  })

  test("numbers different callout kinds independently within a module page", () => {
    const root: Root = {
      type: "root",
      children: [
        callout("theorem", "A theorem", "Theorem"),
        callout("example", "An example", "Example"),
      ],
    }

    process(root)

    assert.match(toString(root.children[0]), /Theorem 1\.1\s\(A theorem\)\./)
    assert.match(toString(root.children[1]), /Example 1\.1\s\(An example\)\./)
  })

  test("falls back to page-local numbering when the slug has no module number", () => {
    const root: Root = {
      type: "root",
      children: [callout("theorem", "A theorem", "Theorem")],
    }

    process(root, "notes/probability")

    assert.match(toString(root), /Theorem 1\s\(A theorem\)\./)
  })

  test("resolves references to internal links", () => {
    const root: Root = {
      type: "root",
      children: [
        callout("theorem", "Change of Variables", "Theorem {#thm:change-vars}"),
        paragraph("See @thm:change-vars."),
      ],
    }

    process(root)

    const link = firstElement(root, (node) => node.tagName === "a")
    assert.equal(link?.properties?.href, "#thm-change-vars")
    assert.deepEqual(link?.properties?.className, ["theorem-ref", "internal"])
    assert.equal(toString(link!), "Theorem 1.1")
  })

  test("resolves cross-page references and records outgoing links", () => {
    const sourceRoot: Root = {
      type: "root",
      children: [callout("theorem", "Change of Variables", "Theorem {#thm:change-vars}")],
    }
    const targetRoot: Root = {
      type: "root",
      children: [paragraph("See @thm:change-vars.")],
    }
    const sourceFile = file("modules/01-probability-basics")
    const targetFile = file("modules/02-deep-learning-basics")
    const content: ProcessedContent[] = [
      [sourceRoot, sourceFile],
      [targetRoot, targetFile],
    ]

    markTheoremReferences(sourceRoot, sourceFile)
    markTheoremReferences(targetRoot, targetFile)
    finalizeTheoremReferences(
      { allSlugs: [sourceFile.data.slug!, targetFile.data.slug!] },
      content,
      () => {},
    )

    const link = firstElement(targetRoot, (node) => node.tagName === "a")
    assert.equal(link?.properties?.href, "../modules/01-probability-basics#thm-change-vars")
    assert.deepEqual(targetFile.data.links, ["modules/01-probability-basics"])
  })

  test("marks unresolved references and emits a warning", () => {
    const root: Root = { type: "root", children: [paragraph("See @thm:missing.")] }
    const vfile = file("modules/01-probability-basics")
    const warnings: string[] = []

    markTheoremReferences(root, vfile)
    finalizeTheoremReferences({ allSlugs: [vfile.data.slug!] }, [[root, vfile]], (msg) =>
      warnings.push(msg),
    )

    const unresolved = firstElement(root, (node) => node.tagName === "span")
    assert.deepEqual(unresolved?.properties?.className, ["theorem-ref", "unresolved"])
    assert.equal(toString(unresolved!), "@thm:missing")
    assert.equal(warnings.length, 1)
  })

  test("does not treat bracketed citations as theorem references", () => {
    const root: Root = { type: "root", children: [paragraph("See [@goodfellow2020generative].")] }
    const vfile = file("modules/04-generative-adversarial-networks")

    markTheoremReferences(root, vfile)
    finalizeTheoremReferences({ allSlugs: [vfile.data.slug!] }, [[root, vfile]], () => {})

    assert.equal(toString(root), "See [@goodfellow2020generative].")
    assert.equal(
      firstElement(root, (node) => node.properties?.["data-theorem-ref"] !== undefined),
      undefined,
    )
  })

  test("throws on duplicate labels", () => {
    const root: Root = {
      type: "root",
      children: [
        callout("theorem", "First", "Theorem {#thm:dup}"),
        callout("theorem", "Second", "Theorem {#thm:dup}"),
      ],
    }
    const vfile = file("modules/01-probability-basics")

    markTheoremReferences(root, vfile)

    assert.throws(
      () => finalizeTheoremReferences({ allSlugs: [vfile.data.slug!] }, [[root, vfile]], () => {}),
      /Duplicate theorem label "thm:dup"/,
    )
  })

  test("finalizer is idempotent", () => {
    const root: Root = {
      type: "root",
      children: [
        callout("theorem", "Change of Variables", "Theorem {#thm:change-vars}"),
        paragraph("See @thm:change-vars."),
      ],
    }
    const vfile = file("modules/01-probability-basics")
    const content: ProcessedContent[] = [[root, vfile]]

    markTheoremReferences(root, vfile)
    finalizeTheoremReferences({ allSlugs: [vfile.data.slug!] }, content, () => {})
    const once = toString(root)
    finalizeTheoremReferences({ allSlugs: [vfile.data.slug!] }, content, () => {})

    assert.equal(toString(root), once)
    assert.equal((toString(root).match(/Theorem 1\.1/g) ?? []).length, 2)
  })
})

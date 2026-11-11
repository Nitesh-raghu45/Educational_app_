import { renderToStaticMarkup } from "react-dom/server";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { describe, expect, it } from "vitest";
import { prepareAnswer, remarkBreakTags, splitMathAtBreaks, stripCitations } from "./markdownPrep";

function render(markdown: string): string {
  return renderToStaticMarkup(
    <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath, remarkBreakTags]} rehypePlugins={[rehypeKatex]}>
      {prepareAnswer(markdown)}
    </ReactMarkdown>,
  );
}

describe("<br> in answers", () => {
  it("splits inline math around <br> so KaTeX never sees the tag", () => {
    expect(splitMathAtBreaks("$6 = 2 \\times 3 <br> 72 = 2^3$")).toBe("$6 = 2 \\times 3$<br>$72 = 2^3$");
    expect(splitMathAtBreaks("$a + b$ and $$x<br>y$$")).toBe("$a + b$ and $$x<br>y$$");
  });

  it("becomes a line break inside table cells, as models write them", () => {
    const html = render(
      "| Step | What is done |\n|---|---|\n| 1. Primes | • $6 = 2 \\times 3$ <br>• $72 = 2^3 \\times 3^2$ <br/>• $120 = 2^3 \\times 3 \\times 5$ |",
    );
    expect(html).not.toContain("&lt;br");
    expect(html.match(/<br\/?>/g)?.length).toBe(2);
    expect(html).toContain("<table>");
    expect(html).toContain("katex");
  });

  it("works in paragraphs, inside math, and leaves other HTML as text", () => {
    const html = render("Line one<br>Line two, $a <br> b$ and <script>alert(1)</script>");
    expect(html).not.toContain("&lt;br");
    expect(html).not.toContain("<script>");
  });

  it("is a newline in copied text", () => {
    expect(stripCitations("a<br>b [1]")).toBe("a\nb");
  });
});

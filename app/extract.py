"""Text extraction for DOCX and EPUB (plus encoding-aware TXT / MD).

Both formats are zip archives of XML, so the standard library is enough:
no new dependency, and the file is parsed on this machine like everything
else. The output is plain text with blank lines between paragraphs, which is
exactly what the chunker splits on.

Archives are read member by member with hard caps, never unpacked to disk.
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from html.parser import HTMLParser
from pathlib import PurePosixPath
from urllib.parse import unquote
from xml.etree import ElementTree as ET

SUPPORTED_EXTENSIONS = (".docx", ".epub", ".txt", ".md")
# One XML part / chapter, and everything read from one archive.
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 96 * 1024 * 1024
MAX_MEMBERS = 5000

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
OPF = "{http://www.idpf.org/2007/opf}"
CONTAINER = "{urn:oasis:names:tc:opendocument:xmlns:container}"


class ExtractError(ValueError):
    """The file cannot be turned into text; the message is safe to show."""


def extension_of(filename: str) -> str:
    return PurePosixPath((filename or "").replace("\\", "/")).suffix.lower()


def extract_text(filename: str, data: bytes) -> str:
    ext = extension_of(filename)
    if ext == ".docx":
        return docx_to_text(data)
    if ext == ".epub":
        return epub_to_text(data)
    if ext in (".txt", ".md"):
        return decode_text(data)
    raise ExtractError("暂只支持 DOCX / EPUB / TXT / MD（PDF 由浏览器端解析）")


def decode_text(data: bytes) -> str:
    """UTF-8 first; Chinese .txt files are still often GBK, so try that next."""
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# zip access with limits
# ---------------------------------------------------------------------------


class _Archive:
    def __init__(self, data: bytes):
        try:
            self.zip = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as e:
            raise ExtractError("文件已损坏或不是有效的文档压缩包") from e
        self.names = set(self.zip.namelist())
        if len(self.names) > MAX_MEMBERS:
            raise ExtractError("文档内部文件数量异常，已拒绝解析")
        self.budget = MAX_TOTAL_BYTES

    def has(self, name: str) -> bool:
        return name in self.names

    def read(self, name: str) -> bytes:
        """Read one member with a streamed cap, so declared sizes are not trusted."""
        try:
            with self.zip.open(name) as handle:
                blob = handle.read(MAX_MEMBER_BYTES + 1)
        except KeyError as e:
            raise ExtractError(f"文档缺少必要部分：{name}") from e
        except (zipfile.BadZipFile, RuntimeError, OSError) as e:
            raise ExtractError("文档内容无法读取（可能已加密或损坏）") from e
        if len(blob) > MAX_MEMBER_BYTES:
            raise ExtractError("文档单个部分过大，已拒绝解析")
        self.budget -= len(blob)
        if self.budget < 0:
            raise ExtractError("文档解压后过大，已拒绝解析")
        return blob


def _parse_xml(blob: bytes, what: str) -> ET.Element:
    try:
        return ET.fromstring(blob)
    except ET.ParseError as e:
        raise ExtractError(f"{what} 不是有效的 XML") from e


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

# Subtrees that are not part of the text a reader sees: tracked deletions,
# moved-away text, field instructions, and the fallback copy of a drawing
# (its primary copy sits in mc:Choice, so reading both would duplicate it).
_DOCX_SKIP = {f"{W}del", f"{W}moveFrom", f"{W}delText", f"{W}instrText", f"{MC}Fallback"}
_DOCX_TEXT = {f"{W}t", f"{M}t"}
_DOCX_BREAK = {f"{W}br", f"{W}cr"}


def docx_to_text(data: bytes) -> str:
    archive = _Archive(data)
    if not archive.has("word/document.xml"):
        raise ExtractError("不是有效的 Word 文档（缺少 word/document.xml）")
    root = _parse_xml(archive.read("word/document.xml"), "Word 正文")
    body = root.find(f"{W}body")
    if body is None:
        raise ExtractError("Word 文档没有正文")
    blocks = [block for block in _docx_blocks(body) if block.strip()]
    return "\n\n".join(blocks)


def _docx_blocks(container: ET.Element):
    for child in container:
        if child.tag == f"{W}p":
            yield _docx_paragraph(child)
        elif child.tag == f"{W}tbl":
            yield _docx_table(child)
        elif child.tag == f"{W}sdt":  # content control (tables of contents, cover blocks)
            content = child.find(f"{W}sdtContent")
            if content is not None:
                yield from _docx_blocks(content)


def _docx_paragraph(paragraph: ET.Element) -> str:
    parts: list[str] = []

    def walk(element: ET.Element) -> None:
        tag = element.tag
        if tag in _DOCX_SKIP:
            return
        if tag in _DOCX_TEXT:
            parts.append(element.text or "")
            return
        if tag == f"{W}tab":
            parts.append("\t")
        elif tag in _DOCX_BREAK:
            parts.append("\n")
        for child in element:
            walk(child)

    walk(paragraph)
    return "".join(parts).strip()


def _docx_table(table: ET.Element) -> str:
    rows = []
    for row in table.findall(f"{W}tr"):  # direct rows only: nested tables come via the cell
        cells = []
        for cell in row.findall(f"{W}tc"):
            text = " ".join(block for block in _docx_blocks(cell) if block.strip())
            cells.append(" ".join(text.split()))
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# EPUB
# ---------------------------------------------------------------------------

_BLOCK_TAGS = {
    "p", "div", "section", "article", "blockquote", "pre", "li", "ul", "ol", "dl", "dt", "dd",
    "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "header", "footer", "figure",
    "figcaption", "hr", "aside", "body",
}  # fmt: skip
_SKIP_TAGS = {"script", "style", "head", "title", "nav", "svg"}
_HTML_TYPES = {"application/xhtml+xml", "text/html"}


class _HtmlText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self.skip_depth += 1
        elif tag == "br":
            self.parts.append("\n")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_startendtag(self, tag, attrs):
        if tag == "br":
            self.parts.append("\n")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    parser = _HtmlText()
    parser.feed(markup)
    parser.close()
    paragraphs = []
    for block in re.split(r"\n\s*\n", "".join(parser.parts)):
        lines = [" ".join(line.split()) for line in block.splitlines()]
        text = "\n".join(line for line in lines if line).strip(" |\n")
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def epub_to_text(data: bytes) -> str:
    archive = _Archive(data)
    if not archive.has("META-INF/container.xml"):
        raise ExtractError("不是有效的 EPUB（缺少 META-INF/container.xml）")
    container = _parse_xml(archive.read("META-INF/container.xml"), "EPUB 容器")
    rootfile = container.find(f".//{CONTAINER}rootfile")
    opf_path = (rootfile.get("full-path") if rootfile is not None else "") or ""
    if not opf_path or not archive.has(opf_path):
        raise ExtractError("EPUB 缺少内容清单（OPF）")
    opf = _parse_xml(archive.read(opf_path), "EPUB 内容清单")
    base = posixpath.dirname(opf_path)

    manifest: dict[str, tuple[str, str, str]] = {}
    for item in opf.iter(f"{OPF}item"):
        manifest[item.get("id", "")] = (
            item.get("href", ""),
            item.get("media-type", ""),
            item.get("properties", ""),
        )
    chapters: list[str] = []
    # The spine is the reading order; the manifest order is arbitrary.
    for ref in opf.iter(f"{OPF}itemref"):
        href, media_type, properties = manifest.get(ref.get("idref", ""), ("", "", ""))
        if not href or media_type not in _HTML_TYPES:
            continue
        if "nav" in properties.split():  # the table of contents is not content
            continue
        name = posixpath.normpath(posixpath.join(base, unquote(href.split("#", 1)[0])))
        if name.startswith("..") or not archive.has(name):
            continue
        text = html_to_text(decode_text(archive.read(name)))
        if text:
            chapters.append(text)
    if not chapters:
        raise ExtractError("EPUB 中没有可读取的正文章节")
    return "\n\n".join(chapters)

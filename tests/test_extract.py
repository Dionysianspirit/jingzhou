"""DOCX / EPUB text extraction.

Fixtures are built in memory with zipfile and hand-written XML, so the tests
need neither python-docx nor real books.
"""

import base64
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from urllib.parse import unquote

import numpy as np
from fastapi.testclient import TestClient

import app.main as main_mod
from app import extract
from app.extract import ExtractError, decode_text, docx_to_text, epub_to_text, extract_text
from app.store import DocStore

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DOC_OPEN = (
    f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<w:document xmlns:w="{W_NS}" '
    f'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
    f'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"><w:body>'
)
DOC_CLOSE = "<w:sectPr/></w:body></w:document>"


def p(text: str) -> str:
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


def make_docx(body: str, extra: dict | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", DOC_OPEN + body + DOC_CLOSE)
        for name, content in (extra or {}).items():
            zf.writestr(name, content)
    return buf.getvalue()


def make_epub(chapters: dict, spine: list, *, opf_dir="OEBPS", manifest_extra="") -> bytes:
    """chapters: {item id: (href, xhtml)}; spine: item ids in reading order."""
    items = "".join(
        f'<item id="{cid}" href="{href}" media-type="application/xhtml+xml"/>'
        for cid, (href, _) in chapters.items()
    )
    refs = "".join(f'<itemref idref="{cid}"/>' for cid in spine)
    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        f"<manifest>{items}{manifest_extra}</manifest><spine>{refs}</spine></package>"
    )
    container = (
        '<?xml version="1.0"?>'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        f'<rootfiles><rootfile full-path="{opf_dir}/content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", container)
        zf.writestr(f"{opf_dir}/content.opf", opf)
        for href, markup in chapters.values():
            zf.writestr(f"{opf_dir}/{unquote(href.split('#', 1)[0])}", markup)
    return buf.getvalue()


def xhtml(body: str, head: str = "") -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?><!DOCTYPE html>'
        f'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>忽略我</title>{head}</head>'
        f"<body>{body}</body></html>"
    )


class DocxTests(unittest.TestCase):
    def test_paragraphs_become_blank_line_separated_blocks(self):
        text = docx_to_text(make_docx(p("第一章 向量") + p("向量加法按分量进行。") + "<w:p/>" + p("第二段。")))
        self.assertEqual(text, "第一章 向量\n\n向量加法按分量进行。\n\n第二段。")

    def test_runs_tabs_and_breaks_inside_one_paragraph(self):
        body = (
            "<w:p><w:r><w:t>秩</w:t></w:r><w:r><w:tab/><w:t>rank</w:t></w:r>"
            '<w:r><w:br/><w:t xml:space="preserve"> 第二行</w:t></w:r></w:p>'
        )
        self.assertEqual(docx_to_text(make_docx(body)), "秩\trank\n 第二行")

    def test_tracked_deletions_are_dropped_and_insertions_kept(self):
        body = (
            "<w:p><w:r><w:t>结论</w:t></w:r>"
            '<w:del w:id="1"><w:r><w:delText>非常非常</w:delText></w:r></w:del>'
            '<w:ins w:id="2"><w:r><w:t>较为</w:t></w:r></w:ins>'
            "<w:r><w:t>有效。</w:t></w:r></w:p>"
        )
        self.assertEqual(docx_to_text(make_docx(body)), "结论较为有效。")

    def test_tables_are_read_row_by_row(self):
        body = (
            p("表 1 结果")
            + "<w:tbl><w:tr><w:tc>" + p("模型") + "</w:tc><w:tc>" + p("准确率") + "</w:tc></w:tr>"
            + "<w:tr><w:tc>" + p("本文方法") + "</w:tc><w:tc>" + p("81%") + "</w:tc></w:tr></w:tbl>"
        )
        self.assertEqual(docx_to_text(make_docx(body)), "表 1 结果\n\n模型 | 准确率\n本文方法 | 81%")

    def test_nested_table_rows_are_not_duplicated(self):
        inner = "<w:tbl><w:tr><w:tc>" + p("内层") + "</w:tc></w:tr></w:tbl>"
        body = "<w:tbl><w:tr><w:tc>" + p("外层") + inner + "</w:tc></w:tr></w:tbl>"
        text = docx_to_text(make_docx(body))
        self.assertEqual(text.count("内层"), 1)
        self.assertIn("外层", text)

    def test_content_controls_are_entered(self):
        body = "<w:sdt><w:sdtPr/><w:sdtContent>" + p("目录项") + "</w:sdtContent></w:sdt>" + p("正文")
        self.assertEqual(docx_to_text(make_docx(body)), "目录项\n\n正文")

    def test_text_box_fallback_copy_is_not_read_twice(self):
        box = "<w:txbxContent>" + p("文本框内容") + "</w:txbxContent>"
        body = (
            "<w:p><w:r><mc:AlternateContent>"
            f"<mc:Choice Requires=\"wps\">{box}</mc:Choice><mc:Fallback>{box}</mc:Fallback>"
            "</mc:AlternateContent></w:r></w:p>"
        )
        self.assertEqual(docx_to_text(make_docx(body)), "文本框内容")

    def test_office_math_text_and_field_codes(self):
        body = (
            "<w:p><w:r><w:t>设 </w:t></w:r><m:oMath><m:r><m:t>A x = b</m:t></m:r></m:oMath>"
            "<w:r><w:instrText> PAGEREF _Toc1 </w:instrText></w:r></w:p>"
        )
        self.assertEqual(docx_to_text(make_docx(body)), "设 A x = b")

    def test_not_a_zip_or_missing_body_is_a_clear_error(self):
        with self.assertRaises(ExtractError):
            docx_to_text(b"this is not a docx")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("hello.txt", "x")
        with self.assertRaises(ExtractError) as ctx:
            docx_to_text(buf.getvalue())
        self.assertIn("word/document.xml", str(ctx.exception))
        with self.assertRaises(ExtractError):
            docx_to_text(make_docx("")[:-40])  # truncated archive

    def test_malformed_xml_is_a_clear_error(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("word/document.xml", "<w:document><unclosed>")
        with self.assertRaises(ExtractError):
            docx_to_text(buf.getvalue())

    def test_oversized_part_is_refused(self):
        with patch.object(extract, "MAX_MEMBER_BYTES", 200):
            with self.assertRaises(ExtractError) as ctx:
                docx_to_text(make_docx(p("字" * 500)))
        self.assertIn("过大", str(ctx.exception))


class EpubTests(unittest.TestCase):
    def test_chapters_follow_the_spine_not_the_manifest(self):
        data = make_epub(
            {
                "c2": ("ch2.xhtml", xhtml("<h1>第二章</h1><p>矩阵。</p>")),
                "c1": ("ch1.xhtml", xhtml("<h1>第一章</h1><p>向量。</p>")),
            },
            spine=["c1", "c2"],
        )
        self.assertEqual(epub_to_text(data), "第一章\n\n向量。\n\n第二章\n\n矩阵。")

    def test_markup_is_reduced_to_readable_text(self):
        body = (
            "<h2>秩</h2><p>秩 &amp; 零化度：rank(A)&#160;+ dim&nbsp;ker(A) = n。<br/>第二行</p>"
            "<ul><li>要点一</li><li>要点二</li></ul>"
            "<script>alert('不该出现')</script><style>p{color:red}</style>"
            "<table><tr><th>模型</th><th>准确率</th></tr><tr><td>本文</td><td>81%</td></tr></table>"
        )
        text = epub_to_text(make_epub({"c1": ("ch1.xhtml", xhtml(body))}, spine=["c1"]))
        self.assertIn("秩 & 零化度：rank(A) + dim ker(A) = n。\n第二行", text)
        self.assertIn("要点一\n\n要点二", text)
        self.assertIn("模型 | 准确率", text)
        self.assertIn("本文 | 81%", text)
        self.assertNotIn("不该出现", text)
        self.assertNotIn("color", text)
        self.assertNotIn("忽略我", text)  # <title> lives in <head>

    def test_navigation_document_and_non_html_items_are_skipped(self):
        extra = (
            '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
            '<item id="css" href="s.css" media-type="text/css"/>'
        )
        data = make_epub(
            {"c1": ("ch1.xhtml", xhtml("<p>正文。</p>")), "nav": ("nav.xhtml", xhtml("<p>目录</p>"))},
            spine=["nav", "css", "c1", "ghost"],
            manifest_extra=extra,
        )
        self.assertEqual(epub_to_text(data), "正文。")

    def test_percent_encoded_and_nested_hrefs_resolve(self):
        data = make_epub(
            {"c1": ("text/%E7%AC%AC%E4%B8%80%E7%AB%A0.xhtml#top", xhtml("<p>第一章正文。</p>"))},
            spine=["c1"],
            opf_dir="EPUB/package",
        )
        self.assertEqual(epub_to_text(data), "第一章正文。")

    def test_href_escaping_the_archive_is_ignored(self):
        data = make_epub(
            {"c1": ("../../../etc/passwd", xhtml("<p>x</p>")), "c2": ("ok.xhtml", xhtml("<p>正文。</p>"))},
            spine=["c1", "c2"],
        )
        self.assertEqual(epub_to_text(data), "正文。")

    def test_missing_container_or_empty_book_is_a_clear_error(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
        with self.assertRaises(ExtractError):
            epub_to_text(buf.getvalue())
        with self.assertRaises(ExtractError) as ctx:
            epub_to_text(make_epub({"c1": ("ch1.xhtml", xhtml(""))}, spine=["c1"]))
        self.assertIn("没有可读取的正文", str(ctx.exception))


class DispatchTests(unittest.TestCase):
    def test_dispatch_by_extension_is_case_insensitive(self):
        self.assertEqual(extract_text("讲义.DOCX", make_docx(p("正文"))), "正文")
        self.assertEqual(extract_text("C:\\课件\\笔记.md", "# 标题".encode()), "# 标题")

    def test_gbk_text_files_are_decoded(self):
        self.assertEqual(decode_text("线性代数".encode("gbk")), "线性代数")
        self.assertEqual(decode_text("\ufeff线性代数".encode("utf-8")), "线性代数")

    def test_unsupported_extension(self):
        with self.assertRaises(ExtractError):
            extract_text("slides.pptx", b"x")


async def fake_encode(loop, texts):
    out = np.ones((len(texts), 4), dtype=np.float32)
    return out / 2.0


class IndexFileApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = DocStore(Path(self.tmp.name))
        self.patches = [
            patch.object(main_mod, "store", self.store),
            patch("app.embeddings.encode_async", fake_encode),
        ]
        for p_ in self.patches:
            p_.start()
        self.client = TestClient(main_mod.app)

    def tearDown(self):
        for p_ in self.patches:
            p_.stop()
        self.tmp.cleanup()

    def post(self, filename, data, doc_id="doc-1"):
        return self.client.post(
            "/api/index-file",
            json={
                "doc_id": doc_id,
                "name": "讲义",
                "filename": filename,
                "data_b64": base64.b64encode(data).decode("ascii"),
            },
        )

    def test_docx_is_extracted_indexed_and_readable_back(self):
        resp = self.post("讲义.docx", make_docx(p("秩-零化度定理：rank(A) + dim ker(A) = n。")))
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["format"], "docx")
        self.assertEqual(resp.json()["chunks"], 1)
        chunk = self.client.get("/api/docs/doc-1/chunks/0").json()
        self.assertIn("秩-零化度定理", chunk["text"])

    def test_epub_is_extracted_and_indexed(self):
        data = make_epub({"c1": ("ch1.xhtml", xhtml("<p>特征值与特征向量。</p>"))}, spine=["c1"])
        resp = self.post("教材.epub", data)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["format"], "epub")

    def test_bad_inputs_are_400_with_a_reason(self):
        self.assertEqual(self.post("a.docx", b"not a zip").status_code, 400)
        self.assertEqual(self.post("a.pptx", b"x").status_code, 400)
        empty = self.post("空.docx", make_docx("<w:p/>"))
        self.assertEqual(empty.status_code, 400)
        self.assertIn("未能从文件中提取到文字", empty.json()["detail"])
        bad_b64 = self.client.post(
            "/api/index-file",
            json={"doc_id": "d", "name": "n", "filename": "a.docx", "data_b64": "@@not-base64@@"},
        )
        self.assertEqual(bad_b64.status_code, 400)
        self.assertEqual(self.post("a.docx", make_docx(p("x")), doc_id="../x").status_code, 422)
        self.assertEqual(self.client.get("/api/docs").json(), [])


if __name__ == "__main__":
    unittest.main()

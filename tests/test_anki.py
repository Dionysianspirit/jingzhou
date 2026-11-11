"""Anki export: package structure, text format, stable ids, and the endpoint.

The structural tests need only the standard library. The last class imports
the files with Anki's own importer; it runs when the ``anki`` package is
installed (``pip install anki``) and is skipped otherwise.
"""

import importlib.util
import io
import json
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path
from urllib.parse import unquote

from fastapi.testclient import TestClient

import app.main as main_mod
from app.anki import (
    MODEL_ID,
    clean_cards,
    clean_deck,
    deck_id_for,
    field_html,
    note_guid,
    to_apkg,
    to_tsv,
)

CARDS = [
    {"q": "秩-零化度定理在说什么？", "a": "rank(A) + dim ker(A) = n。\n秩与零化度之和等于列数。"},
    {"q": "特征向量满足什么关系？", "a": "A v = λ v 且 v ≠ 0，其中 a < b & c > d。"},
]


def open_collection(package: bytes):
    """The SQLite collection inside an .apkg, opened from a temp file."""
    tmp = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(io.BytesIO(package)) as zf:
        names = set(zf.namelist())
        path = Path(tmp.name) / "collection.anki2"
        path.write_bytes(zf.read("collection.anki2"))
        media = zf.read("media").decode()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return tmp, conn, names, media


class CleaningTests(unittest.TestCase):
    def test_blank_sides_and_repeated_questions_are_dropped(self):
        cards = [
            {"q": "  问一  ", "a": "答一"},
            {"q": "问一", "a": "重复的问题"},
            {"q": "", "a": "没有问题"},
            {"q": "没有答案", "a": "   "},
            "not a card",
            {"question": "别名字段", "answer": "也接受"},
        ]
        self.assertEqual(clean_cards(cards), [("问一", "答一"), ("别名字段", "也接受")])
        self.assertEqual(clean_cards(None), [])

    def test_deck_name_is_sanitized_but_keeps_subdecks(self):
        self.assertEqual(clean_deck("径舟::线性代数"), "径舟::线性代数")
        self.assertEqual(clean_deck('  我的"卡组"\t\n '), "我的卡组")
        self.assertEqual(clean_deck(""), "径舟")
        self.assertEqual(len(clean_deck("长" * 300)), 80)

    def test_fields_are_escaped_so_text_is_never_read_as_markup(self):
        self.assertEqual(field_html("a < b & c > d\n第二行"), "a &lt; b &amp; c &gt; d<br>第二行")
        self.assertEqual(field_html("<script>x</script>"), "&lt;script&gt;x&lt;/script&gt;")

    def test_ids_are_stable_per_deck_and_question(self):
        self.assertEqual(note_guid("甲", "问"), note_guid("甲", "问"))
        self.assertNotEqual(note_guid("甲", "问"), note_guid("乙", "问"))
        self.assertNotEqual(note_guid("甲", "问"), note_guid("甲", "另一问"))
        self.assertEqual(deck_id_for("径舟"), deck_id_for("径舟"))
        self.assertEqual(len(str(deck_id_for("径舟"))), 13)


class ApkgTests(unittest.TestCase):
    def test_package_holds_a_collection_and_a_media_map(self):
        tmp, conn, names, media = open_collection(to_apkg(CARDS, "径舟::线性代数"))
        self.addCleanup(tmp.cleanup)
        self.addCleanup(conn.close)
        self.assertEqual(names, {"collection.anki2", "media"})
        self.assertEqual(json.loads(media), {})

    def test_one_note_and_one_card_per_flashcard_in_the_named_deck(self):
        tmp, conn, _, _ = open_collection(to_apkg(CARDS, "径舟::线性代数"))
        self.addCleanup(tmp.cleanup)
        self.addCleanup(conn.close)
        notes = conn.execute("SELECT * FROM notes ORDER BY id").fetchall()
        cards = conn.execute("SELECT * FROM cards ORDER BY due").fetchall()
        self.assertEqual(len(notes), 2)
        self.assertEqual(len(cards), 2)
        front, back = notes[0]["flds"].split("\x1f")
        self.assertEqual(front, "秩-零化度定理在说什么？")
        self.assertEqual(back, "rank(A) + dim ker(A) = n。<br>秩与零化度之和等于列数。")
        self.assertIn("a &lt; b &amp; c &gt; d", notes[1]["flds"])
        self.assertEqual({n["mid"] for n in notes}, {MODEL_ID})
        self.assertEqual(notes[0]["guid"], note_guid("径舟::线性代数", "秩-零化度定理在说什么？"))

        col = conn.execute("SELECT models, decks FROM col").fetchone()
        decks = json.loads(col["decks"])
        deck_id = deck_id_for("径舟::线性代数")
        self.assertEqual(decks[str(deck_id)]["name"], "径舟::线性代数")
        self.assertEqual({c["did"] for c in cards}, {deck_id})
        self.assertEqual([c["nid"] for c in cards], [n["id"] for n in notes])
        self.assertEqual([c["due"] for c in cards], [1, 2])  # new cards keep the export order
        self.assertEqual(len({c["id"] for c in cards} | {n["id"] for n in notes}), 4)

        model = json.loads(col["models"])[str(MODEL_ID)]
        self.assertEqual([f["name"] for f in model["flds"]], ["正面", "背面"])
        self.assertIn("{{正面}}", model["tmpls"][0]["qfmt"])
        self.assertIn("{{背面}}", model["tmpls"][0]["afmt"])

    def test_same_cards_keep_their_guids_across_exports(self):
        def guids(package):
            tmp, conn, _, _ = open_collection(package)
            try:
                return [r[0] for r in conn.execute("SELECT guid FROM notes ORDER BY id")]
            finally:
                conn.close()
                tmp.cleanup()

        first = guids(to_apkg(CARDS, "径舟", now=1_700_000_000))
        later = guids(to_apkg(CARDS, "径舟", now=1_800_000_000))
        self.assertEqual(first, later)  # re-import updates instead of duplicating

    def test_empty_export_is_still_a_valid_package(self):
        tmp, conn, _, _ = open_collection(to_apkg([], "径舟"))
        self.addCleanup(tmp.cleanup)
        self.addCleanup(conn.close)
        self.assertEqual(conn.execute("SELECT count(*) FROM notes").fetchone()[0], 0)


class TsvTests(unittest.TestCase):
    def test_headers_then_question_answer_guid_rows(self):
        lines = to_tsv(CARDS, "径舟::线性代数").splitlines()
        self.assertEqual(
            lines[:4], ["#separator:tab", "#html:true", "#deck:径舟::线性代数", "#guid column:3"]
        )
        rows = [line.split("\t") for line in lines[4:]]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(len(row) == 3 for row in rows))
        self.assertEqual(rows[0][0], "秩-零化度定理在说什么？")
        self.assertEqual(rows[0][1], "rank(A) + dim ker(A) = n。<br>秩与零化度之和等于列数。")
        self.assertEqual(rows[0][2], note_guid("径舟::线性代数", "秩-零化度定理在说什么？"))

    def test_columns_are_left_unnamed(self):
        """Named columns are matched against field names, which differ per Anki language."""
        self.assertNotIn("#columns", to_tsv(CARDS))
        self.assertNotIn("#notetype", to_tsv(CARDS))

    def test_tabs_and_newlines_inside_a_field_cannot_break_the_row(self):
        text = to_tsv([{"q": "含\t制表符\n和换行", "a": "答"}])
        row = text.splitlines()[-1].split("\t")
        self.assertEqual(len(row), 3)
        self.assertEqual(row[0], "含 制表符<br>和换行")


class ExportApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main_mod.app)

    def test_apkg_download(self):
        resp = self.client.post(
            "/api/flashcards/export", json={"cards": CARDS, "deck": "径舟::线性代数"}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.headers["x-card-count"], "2")
        disposition = resp.headers["content-disposition"]
        self.assertIn('filename="jingzhou.apkg"', disposition)
        self.assertIn("径舟--线性代数.apkg", unquote(disposition))
        tmp, conn, names, _ = open_collection(resp.content)
        self.addCleanup(tmp.cleanup)
        self.addCleanup(conn.close)
        self.assertIn("collection.anki2", names)
        self.assertEqual(conn.execute("SELECT count(*) FROM notes").fetchone()[0], 2)

    def test_text_download(self):
        resp = self.client.post("/api/flashcards/export", json={"cards": CARDS, "format": "tsv"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(resp.headers["content-type"].startswith("text/tab-separated-values"))
        self.assertTrue(resp.text.startswith("#separator:tab\n"))
        self.assertIn("#deck:径舟\n", resp.text)

    def test_needs_no_llm_key_and_no_documents(self):
        from unittest.mock import patch

        with patch("app.llm.llm", None):
            resp = self.client.post("/api/flashcards/export", json={"cards": CARDS})
        self.assertEqual(resp.status_code, 200)

    def test_nothing_to_export_is_a_clear_400(self):
        resp = self.client.post("/api/flashcards/export", json={"cards": [{"q": " ", "a": "答"}]})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("没有可导出的笺卡", resp.json()["detail"])

    def test_bad_requests_are_rejected(self):
        post = lambda body: self.client.post("/api/flashcards/export", json=body)  # noqa: E731
        self.assertEqual(post({"cards": []}).status_code, 422)
        self.assertEqual(post({"cards": CARDS, "format": "pdf"}).status_code, 422)
        self.assertEqual(post({"cards": [{"q": "只有问题"}]}).status_code, 422)


@unittest.skipUnless(importlib.util.find_spec("anki"), "needs the optional `anki` package")
class RealAnkiImportTests(unittest.TestCase):
    """End to end against Anki's own importer: the files are only useful if
    Anki accepts them and puts the text in the right fields."""

    def setUp(self):
        from anki.collection import Collection

        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.col = Collection(str(self.dir / "collection.anki2"))

    def tearDown(self):
        self.col.close()
        self.tmp.cleanup()

    def notes(self):
        return {n.guid: n for n in map(self.col.get_note, self.col.find_notes(""))}

    def import_apkg(self, package: bytes, name: str):
        from anki.collection import ImportAnkiPackageOptions, ImportAnkiPackageRequest

        path = self.dir / name
        path.write_bytes(package)
        request = ImportAnkiPackageRequest(
            package_path=str(path), options=ImportAnkiPackageOptions()
        )
        return self.col.import_anki_package(request).log

    def import_text(self, text: str):
        from anki import import_export_pb2 as pb

        path = self.dir / "cards.txt"
        path.write_text(text, encoding="utf-8")
        metadata = self.col.get_csv_metadata(path=str(path), delimiter=None)
        return self.col.import_csv(pb.ImportCsvRequest(path=str(path), metadata=metadata)).log

    def test_apkg_imports_then_updates_in_place(self):
        log = self.import_apkg(to_apkg(CARDS, "径舟::线性代数", now=1_700_000_000), "a.apkg")
        self.assertEqual(len(log.new), 2)
        guid = note_guid("径舟::线性代数", CARDS[0]["q"])
        note = self.notes()[guid]
        self.assertEqual(
            note.fields,
            ["秩-零化度定理在说什么？", "rank(A) + dim ker(A) = n。<br>秩与零化度之和等于列数。"],
        )
        self.assertEqual(self.col.decks.name(note.cards()[0].did), "径舟::线性代数")
        self.assertIn("秩-零化度定理在说什么？", note.cards()[0].question())

        revised = [dict(CARDS[0], a="（已修订）"), CARDS[1], {"q": "新卡", "a": "新答"}]
        log = self.import_apkg(to_apkg(revised, "径舟::线性代数", now=1_700_003_600), "b.apkg")
        self.assertEqual(len(log.new), 1)
        self.assertEqual(self.col.note_count(), 3)  # updated, not duplicated
        self.assertEqual(self.notes()[guid].fields[1], "（已修订）")

    def test_text_file_maps_question_and_answer_without_manual_mapping(self):
        log = self.import_text(to_tsv(CARDS, "径舟::线性代数"))
        self.assertEqual(len(log.new), 2)
        note = self.notes()[note_guid("径舟::线性代数", CARDS[1]["q"])]
        self.assertEqual(note.fields[0], "特征向量满足什么关系？")
        self.assertEqual(note.fields[1], "A v = λ v 且 v ≠ 0，其中 a &lt; b &amp; c &gt; d。")
        self.assertEqual(self.col.decks.name(note.cards()[0].did), "径舟::线性代数")
        again = self.import_text(to_tsv(CARDS, "径舟::线性代数"))
        self.assertEqual(len(again.new), 0)
        self.assertEqual(self.col.note_count(), 2)


if __name__ == "__main__":
    unittest.main()

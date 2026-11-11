"""Export flashcards to Anki.

Two formats, both built with the standard library:

- ``.apkg``: a real Anki package (a zip holding an SQLite collection and a
  media map). It carries its own note type and deck, so it imports the same
  way on desktop and mobile with a double click / share.
- ``.txt``: Anki's tab-separated text format with file headers, for people
  who want to look at or edit the cards before importing.

Every note gets a GUID derived from its deck and question. Exporting the same
cards again and re-importing therefore updates them instead of duplicating
them, and review history in Anki is kept.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import sqlite3
import tempfile
import time
import zipfile
from pathlib import Path

DEFAULT_DECK = "径舟"
MAX_CARDS = 500
# A fixed id keeps every export on one shared note type inside Anki.
MODEL_ID = 1731300000101
MODEL_NAME = "径舟 · 问答"
FIELD_SEP = "\x1f"

CARD_CSS = (
    ".card { font-family: 'Noto Serif SC', 'Songti SC', serif; font-size: 20px; "
    "line-height: 1.7; text-align: left; color: #1E2937; background: #F7F5F2; "
    "max-width: 640px; margin: 0 auto; padding: 20px; }\n"
    "#answer { border: 0; border-top: 1px solid rgba(30,41,55,0.15); margin: 18px 0; }"
)

SCHEMA = """
CREATE TABLE col (
    id integer primary key, crt integer not null, mod integer not null,
    scm integer not null, ver integer not null, dty integer not null,
    usn integer not null, ls integer not null, conf text not null,
    models text not null, decks text not null, dconf text not null, tags text not null
);
CREATE TABLE notes (
    id integer primary key, guid text not null, mid integer not null,
    mod integer not null, usn integer not null, tags text not null,
    flds text not null, sfld integer not null, csum integer not null,
    flags integer not null, data text not null
);
CREATE TABLE cards (
    id integer primary key, nid integer not null, did integer not null,
    ord integer not null, mod integer not null, usn integer not null,
    type integer not null, queue integer not null, due integer not null,
    ivl integer not null, factor integer not null, reps integer not null,
    lapses integer not null, left integer not null, odue integer not null,
    odid integer not null, flags integer not null, data text not null
);
CREATE TABLE revlog (
    id integer primary key, cid integer not null, usn integer not null,
    ease integer not null, ivl integer not null, lastIvl integer not null,
    factor integer not null, time integer not null, type integer not null
);
CREATE TABLE graves (usn integer not null, oid integer not null, type integer not null);
CREATE INDEX ix_notes_usn on notes (usn);
CREATE INDEX ix_cards_usn on cards (usn);
CREATE INDEX ix_revlog_usn on revlog (usn);
CREATE INDEX ix_cards_nid on cards (nid);
CREATE INDEX ix_cards_sched on cards (did, queue, due);
CREATE INDEX ix_revlog_cid on revlog (cid);
CREATE INDEX ix_notes_csum on notes (csum);
"""


def clean_cards(cards: list[dict]) -> list[tuple[str, str]]:
    """(question, answer) pairs with both sides present; duplicates by question dropped."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        question = str(card.get("q") or card.get("question") or "").strip()
        answer = str(card.get("a") or card.get("answer") or "").strip()
        if not question or not answer or question in seen:
            continue
        seen.add(question)
        out.append((question, answer))
    return out[:MAX_CARDS]


def clean_deck(deck: str) -> str:
    # "::" is Anki's sub-deck separator and is allowed; control chars and quotes are not
    name = "".join(ch for ch in (deck or "") if ch.isprintable() and ch not in '"\t').strip()
    return name[:80] or DEFAULT_DECK


def field_html(text: str) -> str:
    return html.escape(text, quote=False).replace("\r\n", "\n").replace("\n", "<br>")


def note_guid(deck: str, question: str) -> str:
    return "jz" + hashlib.sha1(f"{deck}{FIELD_SEP}{question}".encode()).hexdigest()[:20]


def deck_id_for(deck: str) -> int:
    """Stable 13-digit id per deck name (Anki ids are epoch-millisecond sized)."""
    digest = int(hashlib.sha1(deck.encode()).hexdigest()[:12], 16)
    return 1_400_000_000_000 + digest % 300_000_000_000


def to_tsv(cards: list[dict], deck: str = DEFAULT_DECK) -> str:
    """Anki text import with file headers (Anki 2.1.54+).

    The layout is the one Anki's own importer maps correctly without the user
    touching the field mapping: question, answer, then the GUID last. Columns
    are deliberately left unnamed, because Anki matches named columns against
    the note type's field names, and those differ per language of the user's
    Anki (Front/Back, 正面/背面, ...). Unnamed columns are mapped by position.
    """
    deck = clean_deck(deck)
    lines = ["#separator:tab", "#html:true", f"#deck:{deck}", "#guid column:3"]
    for question, answer in clean_cards(cards):
        cells = [field_html(question), field_html(answer), note_guid(deck, question)]
        lines.append("\t".join(cell.replace("\t", " ") for cell in cells))
    return "\n".join(lines) + "\n"


def to_apkg(cards: list[dict], deck: str = DEFAULT_DECK, *, now: float | None = None) -> bytes:
    deck = clean_deck(deck)
    pairs = clean_cards(cards)
    stamp = int(now if now is not None else time.time())
    stamp_ms = stamp * 1000
    deck_id = deck_id_for(deck)

    model = {
        "id": MODEL_ID,
        "name": MODEL_NAME,
        "type": 0,
        "mod": stamp,
        "usn": -1,
        "sortf": 0,
        "did": deck_id,
        "tmpls": [
            {
                "name": "问答",
                "ord": 0,
                "qfmt": "{{正面}}",
                "afmt": "{{FrontSide}}<hr id=answer>{{背面}}",
                "bqfmt": "",
                "bafmt": "",
                "did": None,
            }
        ],
        "flds": [
            {"name": name, "ord": ord_, "sticky": False, "rtl": False, "font": "Arial",
             "size": 20, "media": []}
            for ord_, name in enumerate(("正面", "背面"))
        ],  # fmt: skip
        "css": CARD_CSS,
        "latexPre": (
            "\\documentclass[12pt]{article}\n\\special{papersize=3in,5in}\n"
            "\\usepackage[utf8]{inputenc}\n\\usepackage{amssymb,amsmath}\n"
            "\\pagestyle{empty}\n\\setlength{\\parindent}{0in}\n\\begin{document}\n"
        ),
        "latexPost": "\\end{document}",
        "latexsvg": False,
        "req": [[0, "any", [0]]],
        "tags": [],
        "vers": [],
    }

    def deck_json(did: int, name: str) -> dict:
        return {
            "id": did, "name": name, "mod": stamp, "usn": -1, "desc": "", "dyn": 0,
            "conf": 1, "collapsed": False, "extendNew": 0, "extendRev": 50,
            "lrnToday": [0, 0], "newToday": [0, 0], "revToday": [0, 0], "timeToday": [0, 0],
        }  # fmt: skip

    decks = {"1": deck_json(1, "Default"), str(deck_id): deck_json(deck_id, deck)}
    conf = {
        "activeDecks": [1], "curDeck": 1, "newSpread": 0, "collapseTime": 1200,
        "timeLim": 0, "estTimes": True, "dueCounts": True, "curModel": None,
        "nextPos": len(pairs) + 1, "sortType": "noteFld", "sortBackwards": False,
        "addToCur": True,
    }  # fmt: skip
    dconf = {
        "1": {
            "id": 1, "name": "Default", "mod": 0, "usn": 0, "maxTaken": 60, "autoplay": True,
            "timer": 0, "replayq": True, "dyn": False,
            "new": {"bury": True, "delays": [1, 10], "initialFactor": 2500, "ints": [1, 4, 7],
                    "order": 1, "perDay": 20, "separate": True},
            "lapse": {"delays": [10], "leechAction": 0, "leechFails": 8, "minInt": 1, "mult": 0},
            "rev": {"bury": True, "ease4": 1.3, "fuzz": 0.05, "ivlFct": 1, "maxIvl": 36500,
                    "minSpace": 1, "perDay": 100},
        }
    }  # fmt: skip

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "collection.anki2"
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT INTO col VALUES (1,?,?,?,11,0,0,0,?,?,?,?,'{}')",
                (
                    stamp,
                    stamp_ms,
                    stamp_ms,
                    json.dumps(conf),
                    json.dumps({str(MODEL_ID): model}, ensure_ascii=False),
                    json.dumps(decks, ensure_ascii=False),
                    json.dumps(dconf),
                ),
            )
            for position, (question, answer) in enumerate(pairs, start=1):
                note_id = stamp_ms + position
                front = field_html(question)
                checksum = int(hashlib.sha1(question.encode()).hexdigest()[:8], 16)
                conn.execute(
                    "INSERT INTO notes VALUES (?,?,?,?,-1,'',?,?,?,0,'')",
                    (
                        note_id,
                        note_guid(deck, question),
                        MODEL_ID,
                        stamp,
                        front + FIELD_SEP + field_html(answer),
                        question,
                        checksum,
                    ),
                )
                conn.execute(
                    "INSERT INTO cards VALUES (?,?,?,0,?,-1,0,0,?,0,0,0,0,0,0,0,0,'')",
                    (note_id + len(pairs), note_id, deck_id, stamp, position),
                )
            conn.commit()
        finally:
            conn.close()
        blob = db_path.read_bytes()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("collection.anki2", blob)
        package.writestr("media", "{}")
    return buffer.getvalue()

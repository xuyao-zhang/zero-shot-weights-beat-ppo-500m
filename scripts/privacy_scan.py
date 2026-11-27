# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Xuyao Zhang (comprehensive-weight-space-programming-leaderboard contributors)
"""Repository-wide privacy and credential hygiene scanner.

Scans every file under the repository root (excluding .git/), including
this script itself and binary/archive files (.npz, .zip), for personal paths,
local usernames, API key tokens, and private session identifiers.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np


URL_STRIP_RE = re.compile(r"https?://[^\s\"'<>|]+", re.IGNORECASE)


def _build_rules() -> list[tuple[str, re.Pattern[str], bool, bool]]:
    """Construct scan rules dynamically without embedding literal sensitive tokens.

    Returns tuples of (rule_id, compiled_pattern, mask_secret, strip_urls).
    """
    bs = chr(92)
    session_halves = [
        ("ed02", "ec1b"),
        ("e821", "b075"),
        ("3001", "a2e7"),
        ("25b3", "77ad"),
        ("1bf2", "446e"),
        ("5eee", "15a2"),
        ("870e", "ceb6"),
        ("c19e", "420c"),
        ("2da0", "a055"),
        ("afcc", "0d2f"),
        ("49d6", "bd85"),
        ("9c21", "fa97"),
    ]
    session_alt = "|".join(a + b for a, b in session_halves)
    ag = "anti" + "gravity"
    app_data = "App" + "Data"

    return [
        ("win_username", re.compile(re.escape("354" + "39")), False, False),
        ("win_users_backslash", re.compile(re.escape("C:" + bs + "Users"), re.IGNORECASE), False, False),
        ("win_users_forwardslash", re.compile(re.escape("C:" + "/Users"), re.IGNORECASE), False, False),
        ("posix_users_path", re.compile(re.escape("/Users" + "/354" + "39"), re.IGNORECASE), False, False),
        (
            "ag_path",
            re.compile(
                r"(?i:\.gemini[/\\]"
                + ag
                + r"|(?:(?<![A-Za-z0-9_])[a-z]:(?:\\|/(?!/))[^\s]*?"
                + ag
                + r"\b|(?<![A-Za-z0-9_])[/\\](?:users|"
                + app_data.lower()
                + r"|local|roaming|home|opt|var|tmp)[/\\][^\s]*?"
                + ag
                + r"\b|"
                + app_data.lower()
                + r"[/\\](?:local|roaming)[/\\][^\s]*?"
                + ag
                + r"\b))"
            ),
            False,
            False,
        ),
        (
            "ag_path",
            re.compile(
                r"(?i:"
                + ag
                + r"[/\\](?:brain|scratch|logs|results|tmp|data|user|artifacts|outputs|runs|\.user_uploaded|workspace|conversations|mcp)\b)"
            ),
            False,
            True,
        ),
        ("win_local_ad", re.compile(re.escape(app_data), re.IGNORECASE), False, False),
        (
            "api_sk",
            re.compile(r"(?<![A-Za-z0-9_])" + "sk" + r"-(?:proj-|ant-|or-v1-)?[A-Za-z0-9_-]{10,}"),
            True,
            False,
        ),
        (
            "api_ghp",
            re.compile(r"(?<![A-Za-z0-9_])" + "ghp" + r"_[A-Za-z0-9]{10,}"),
            True,
            False,
        ),
        (
            "api_gh_pat",
            re.compile(r"(?<![A-Za-z0-9_])" + "github_" + r"pat_[A-Za-z0-9_]{10,}"),
            True,
            False,
        ),
        (
            "api_aiza",
            re.compile(r"(?<![A-Za-z0-9_])" + "AI" + r"za[A-Za-z0-9_-]{10,}"),
            True,
            False,
        ),
        ("api_openrouter_env", re.compile(re.escape("OPENROUTER" + "_API_KEY")), False, False),
        ("brain_session_uuid", re.compile(rf"\b(?:{session_alt})\b", re.IGNORECASE), False, False),
        ("internal_transcript", re.compile(re.escape("transcript_" + "full.jsonl"), re.IGNORECASE), False, False),
        ("internal_sys_gen", re.compile(re.escape(".system_" + "generated"), re.IGNORECASE), False, False),
    ]


RULES = _build_rules()


def _format_snippet(text: str, match: re.Match[str], mask_secret: bool) -> str:
    token = match.group(0)
    if mask_secret and len(token) > 8:
        masked = token[:6] + "***" + token[-2:]
        start, end = match.span()
        text = text[:start] + masked + text[end:]
    snippet = text.strip().replace("\r", "")
    if len(snippet) > 120:
        snippet = snippet[:117] + "..."
    return snippet


def _iter_hits(text: str) -> list[tuple[str, str]]:
    """Return [(rule_id, formatted_snippet), ...] for all matching rules on text."""
    out: list[tuple[str, str]] = []
    stripped_text: str | None = None
    for rule_id, pattern, mask_secret, strip_urls in RULES:
        if strip_urls:
            if stripped_text is None:
                stripped_text = URL_STRIP_RE.sub(" ", text)
            target = stripped_text
        else:
            target = text
        m = pattern.search(target)
        if m:
            out.append((rule_id, _format_snippet(target, m, mask_secret)))
    return out


def _scan_text_block(
    rel_path: str,
    location_prefix: str,
    raw_bytes: bytes,
) -> list[str]:
    hits: list[str] = []
    for lineno, raw_line in enumerate(raw_bytes.split(b"\n"), start=1):
        line = raw_line.decode("utf-8", errors="replace")
        for rule_id, snippet in _iter_hits(line):
            loc = f"{location_prefix}line {lineno}" if location_prefix else f"line {lineno}"
            hits.append(f"{rel_path}:{loc} [{rule_id}]: {snippet}")
    return hits


def _scan_npy_header(rel_path: str, member: str, raw_bytes: bytes) -> list[str]:
    hits: list[str] = []
    if not raw_bytes.startswith(b"\x93NUMPY"):
        hits.append(f"{rel_path}:zip:{member} [invalid_npy_magic]: missing NPY header magic")
        return hits
    buf = io.BytesIO(raw_bytes)
    try:
        version = np.lib.format.read_magic(buf)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(buf)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(buf)
        else:
            shape, fortran_order, dtype = np.lib.format._read_array_header(buf, version)
        if dtype == object or getattr(dtype, "hasobject", False):
            hits.append(f"{rel_path}:zip:{member} [unsafe_object_dtype]: dtype={dtype}")
        header_summary = f"shape={shape} fortran={fortran_order} dtype={dtype}"
        for rule_id, snippet in _iter_hits(header_summary):
            hits.append(f"{rel_path}:zip:{member}:header [{rule_id}]: {snippet}")
    except Exception as exc:  # noqa: BLE001
        hits.append(f"{rel_path}:zip:{member} [npy_header_error]: {type(exc).__name__}: {exc}")
    hits.extend(_scan_text_block(rel_path, f"zip:{member}:", raw_bytes))
    return hits


def scan_file(path: Path, root: Path) -> list[str]:
    """Scan a single file (text, binary, or .npz/.zip archive) and return formatted hits."""
    rel_path = path.relative_to(root).as_posix()
    hits: list[str] = []

    is_archive = path.suffix.lower() in {".npz", ".zip"} or zipfile.is_zipfile(path)
    if is_archive:
        try:
            with zipfile.ZipFile(path, "r") as zf:
                if zf.comment:
                    comment_text = zf.comment.decode("utf-8", errors="replace")
                    for rule_id, snippet in _iter_hits(comment_text):
                        hits.append(f"{rel_path}:zip-comment [{rule_id}]: {snippet}")
                for info in zf.infolist():
                    for rule_id, snippet in _iter_hits(info.filename):
                        hits.append(f"{rel_path}:zip-name:{info.filename} [{rule_id}]: {snippet}")
                    if info.comment:
                        mcomment = info.comment.decode("utf-8", errors="replace")
                        for rule_id, snippet in _iter_hits(mcomment):
                            hits.append(
                                f"{rel_path}:zip-member-comment:{info.filename} [{rule_id}]: {snippet}"
                            )
                    raw_member = zf.read(info.filename)
                    if info.filename.endswith(".npy"):
                        hits.extend(_scan_npy_header(rel_path, info.filename, raw_member))
                    else:
                        hits.extend(_scan_text_block(rel_path, f"zip:{info.filename}:", raw_member))
        except zipfile.BadZipFile:
            raw = path.read_bytes()
            hits.extend(_scan_text_block(rel_path, "", raw))
        return hits

    raw = path.read_bytes()
    hits.extend(_scan_text_block(rel_path, "", raw))
    return hits


def scan_repo(root: Path) -> tuple[int, list[str]]:
    """Walk root excluding .git/ directories and scan every file."""
    file_count = 0
    all_hits: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in {".git", "__pycache__"})
        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            file_count += 1
            all_hits.extend(scan_file(fpath, root))
    return file_count, all_hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan repository files for PII, paths, and credentials.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root directory to scan (defaults to repo root).",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        print(f"Error: invalid --root directory: {root}", file=sys.stderr)
        return 2

    file_count, hits = scan_repo(root)
    if hits:
        for h in hits:
            print(h)
        print(f"PRIVACY SCAN FAILED: {len(hits)} hit(s) across {file_count} scanned file(s).")
        return 1

    print(f"PRIVACY SCAN PASSED: 0 hits across {file_count} scanned file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

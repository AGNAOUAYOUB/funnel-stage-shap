"""Generate the amendment register from the log, so the two cannot drift apart.

The register is the authoritative index an auditor reads first: one row per
amendment, stating when it was taken, whether it preceded test-partition access,
whether it belongs to the confirmatory or the exploratory arm, whether it still
stands, and where the manuscript reports it.

Only the classification is hand-maintained. The identifiers, dates and titles are
parsed from ``docs/AMENDMENTS.md``, and the script fails if the log contains an
identifier the classification does not cover, or vice versa. A register that
silently omits an amendment is worse than no register.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "docs" / "AMENDMENTS.md"
OUT_MD = ROOT / "docs" / "AMENDMENT_REGISTER.md"
OUT_CSV = ROOT / "reports" / "tables" / "amendment_register.csv"

#: The test partition was first opened at A19, which reports the stage
#: improvement curve on 38,059 test sessions. Everything numbered below that,
#: plus A46 (a prose correction), precedes test access.
FIRST_POST_TEST = 19

# id -> (arm, status, manuscript location)
#   arm: Confirmatory | Exploratory | Implementation (a build decision, not a
#        result), Reporting (a correction to how something is described)
CLASSIFICATION: dict[str, tuple[str, str, str]] = {
    "A1":  ("Implementation", "Active", "Sec. 3 (stage definitions)"),
    "A2":  ("Implementation", "Active", "Sec. 3, Table 1"),
    "A3":  ("Implementation", "Active", "Sec. 3 (features)"),
    "A4":  ("Implementation", "Active", "Sec. 3 (sessionisation)"),
    "A5":  ("Implementation", "Open", "Sec. 3 (data sources)"),
    "A6":  ("Implementation", "Active", "Sec. 3 (cut-points)"),
    "A8":  ("Implementation", "Active", "Sec. 3 (cut-points)"),
    "A9":  ("Implementation", "Active", "Data and code availability"),
    "A10": ("Implementation", "Active", "Sec. 3 (data sources)"),
    "A11": ("Implementation", "Active", "Sec. 3 (splitting)"),
    "A12": ("Implementation", "Active", "Sec. 4 (Dataset A)"),
    "A13": ("Implementation", "Active", "not reported (environment)"),
    "A14": ("Implementation", "Active", "not reported (defect fix)"),
    "A15": ("Implementation", "Active", "Sec. 3 (subsample)"),
    "A16": ("Implementation", "Active", "Sec. 3 (cut-points)"),
    "A17": ("Implementation", "Active", "Sec. 3 (splitting), Sec. 5 (limitations)"),
    "A18": ("Implementation", "Active", "Sec. 3 (calibration), Sec. 4 (sensitivity)"),
    "A19": ("Confirmatory", "Active", "Sec. 4, Table 3 (H1)"),
    "A20": ("Confirmatory", "Superseded by A20b", "Sec. 4, Table 5 (H3)"),
    "A20b": ("Confirmatory", "Active", "Sec. 4, Table 5 (H3)"),
    "A21": ("Implementation", "Active", "not reported (defect fix)"),
    "A22": ("Confirmatory", "Active", "Sec. 4, Table 10 (H4)"),
    "A23": ("Exploratory", "Superseded by A23b", "Sec. 4, Table 10 (H4)"),
    "A23b": ("Exploratory", "Active", "Sec. 4, Table 10 (H4)"),
    "A24": ("Exploratory", "Active", "Sec. 3 (tuning), Sec. 4 (sensitivity)"),
    "A25": ("Exploratory", "Active", "Sec. 4 (sensitivity)"),
    "A26": ("Confirmatory", "Active", "Sec. 4, Table 9"),
    "A27": ("Reporting", "Active", "Sec. 2 (related work)"),
    "A28": ("Reporting", "Active", "Data and code availability"),
    "A29": ("Implementation", "Active", "Data and code availability"),
    "A30": ("Confirmatory", "Active", "Sec. 3 (sequence arm), Sec. 4 (H4)"),
    "A31": ("Reporting", "Active", "Sec. 4, Table 3"),
    "A32": ("Confirmatory", "Active", "Sec. 4 (H1)"),
    "A33": ("Exploratory", "Superseded by A43", "Sec. 4 (RQ4)"),
    "A34": ("Reporting", "Active", "Appendix A"),
    "A35": ("Reporting", "Open", "Data and code availability"),
    "A36": ("Reporting", "Active", "figure captions"),
    "A37": ("Reporting", "Active", "Sec. 4 (computational cost)"),
    "A38": ("Exploratory", "Active", "Sec. 5 (decision support)"),
    "A39": ("Exploratory", "Active", "Sec. 5, Table 14"),
    "A40": ("Exploratory", "Active", "Sec. 5, Table 14"),
    "A41": ("Exploratory", "Active", "Sec. 4, Table 4"),
    "A42": ("Exploratory", "Active", "Sec. 5, Table 15"),
    "A43": ("Exploratory", "Active", "Sec. 4, Table 8 (RQ4)"),
    "A44": ("Exploratory", "Active", "Sec. 4, Table 4"),
    "A45": ("Exploratory", "Active", "Sec. 4, Table 7"),
    "A46": ("Reporting", "Active", "Sec. 3 (seeds)"),
    "A47": ("Reporting", "Active", "Sec. 5, Table 12; data availability"),
}

#: Amendments that close an earlier open item, so the register can show the
#: earlier entry as resolved rather than perpetually open.
CLOSURES = {"A5": "A47", "A35": "A47"}


def parse_log() -> list[dict[str, str]]:
    text = io.open(LOG, encoding="utf-8").read()
    entries, date = [], "unknown"
    for line in text.splitlines():
        heading = re.match(r"^## (\d{4}-\d{2}-\d{2})", line)
        if heading:
            date = heading.group(1)
            continue
        entry = re.match(r"^### (A\d+[a-z]?)\.\s*(.+)$", line)
        if entry:
            entries.append(
                {"id": entry.group(1), "date": date, "title": entry.group(2).strip()}
            )
    return entries


def sort_key(identifier: str) -> tuple[int, str]:
    m = re.match(r"A(\d+)([a-z]?)", identifier)
    return int(m.group(1)), m.group(2)


def main() -> int:
    entries = parse_log()
    found = {e["id"] for e in entries}
    classified = set(CLASSIFICATION)

    problems = []
    if found - classified:
        problems.append("in log but unclassified: %s" % sorted(found - classified))
    if classified - found:
        problems.append("classified but not in log: %s" % sorted(classified - found))
    duplicates = [i for i in found if [e["id"] for e in entries].count(i) > 1]
    if duplicates:
        problems.append("duplicate identifiers: %s" % sorted(set(duplicates)))
    if problems:
        for p in problems:
            print("REGISTER MISMATCH:", p, file=sys.stderr)
        return 1

    rows = []
    for e in sorted(entries, key=lambda e: sort_key(e["id"])):
        arm, status, location = CLASSIFICATION[e["id"]]
        if e["id"] in CLOSURES:
            status = "Closed by %s" % CLOSURES[e["id"]]
        number = sort_key(e["id"])[0]
        before_after = (
            "before" if (number < FIRST_POST_TEST or e["id"] == "A46") else "after"
        )
        title = re.sub(r"\s*\(Sec\..*?\)\s*$", "", e["title"])
        title = re.sub(r"\*\*", "", title).strip()
        rows.append({
            "id": e["id"], "date": e["date"], "decision": title,
            "test_access": before_after, "arm": arm, "status": status,
            "manuscript_location": location,
        })

    header = (
        "# Amendment register\n\n"
        "Generated by `scripts/build_amendment_register.py` from `docs/AMENDMENTS.md`.\n"
        "Do not edit by hand: the generator fails if the log and the classification\n"
        "disagree, which is the only thing keeping the two in step.\n\n"
        "**Test access** records whether the decision was taken before or after the test\n"
        "partition was first opened, which happened at A19. **Arm** separates decisions\n"
        "that built the artifact (*Implementation*), decisions that report a pre-specified\n"
        "test (*Confirmatory*), analyses added after the protocol was frozen\n"
        "(*Exploratory*), and corrections to how something was described rather than to\n"
        "what was done (*Reporting*).\n\n"
        "A `b` suffix marks a correction superseding the entry of the same number; both are\n"
        "retained. There is no A7. A46 was originally written as a second A30.\n\n"
        "| ID | Date | Decision | Test access | Arm | Status | Manuscript location |\n"
        "| -- | ---- | -------- | ----------- | --- | ------ | ------------------- |\n"
    )
    body = "".join(
        "| {id} | {date} | {decision} | {test_access} | {arm} | {status} | {manuscript_location} |\n".format(**r)
        for r in rows
    )
    counts = {}
    for r in rows:
        counts[r["arm"]] = counts.get(r["arm"], 0) + 1
    summary = (
        "\n## Counts\n\n"
        "%d amendments in total (A1-A%d, no A7, plus corrections A20b and A23b).\n\n"
        % (len(rows), max(sort_key(r["id"])[0] for r in rows))
        + "".join("- %s: %d\n" % (k, counts[k]) for k in sorted(counts))
        + "- taken before test access: %d\n" % sum(1 for r in rows if r["test_access"] == "before")
        + "- taken after test access: %d\n" % sum(1 for r in rows if r["test_access"] == "after")
        + "- superseded or open: %d\n" % sum(1 for r in rows if r["status"] != "Active")
    )
    io.open(OUT_MD, "w", encoding="utf-8", newline="\n").write(header + body + summary)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with io.open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("%d amendments -> %s" % (len(rows), OUT_MD))
    print("%d amendments -> %s" % (len(rows), OUT_CSV))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

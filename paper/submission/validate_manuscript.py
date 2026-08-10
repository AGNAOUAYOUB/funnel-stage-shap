"""Manuscript validation. Written via Write, not a heredoc: a bash heredoc
collapses \\\\ to \\, which silently turns regexes like \\\\ref into \\r
(carriage return) and makes the check pass by matching nothing."""

import os
import re

TEX = r"C:\Users\Pc\Desktop\dynamic_xai\paper\submission\main.tex"
BIB = r"C:\Users\Pc\Desktop\dynamic_xai\paper\submission\references.bib"
FIGDIR = r"C:\Users\Pc\Desktop\dynamic_xai\paper\figures"

tex = open(TEX, encoding="utf-8").read()
bib = open(BIB, encoding="utf-8").read()

BS = chr(92)  # backslash, kept out of string literals to avoid escaping doubt


def rx(pattern):
    return pattern.replace("@", BS + BS)


# ---- sanity: the parser must actually see the document ----
n_sections = len(re.findall(rx(r"@section\{"), tex))
n_labels = len(re.findall(rx(r"@label\{"), tex))
n_refs = len(re.findall(rx(r"@ref\{"), tex))
print("PARSER SANITY  sections=%d labels=%d refs=%d" % (n_sections, n_labels, n_refs))
assert n_sections > 0 and n_labels > 0 and n_refs > 0, "parser sees nothing - check escaping"

# ---- stray control characters ----
# The same heredoc collapse that this module's docstring warns about produced a
# literal CR where a `\ref` belonged, which typeset as "Table eftab:sweep" and
# was invisible to every check below, since none of them look for a `\ref` that
# is no longer a `\ref`. A byte-level scan is the only thing that catches it.
raw = open(TEX, "rb").read()
control = {b: raw.count(bytes([b])) for b in range(32) if b != 10 and bytes([b]) in raw}
if control:
    names = {9: "TAB", 13: "CR", 12: "FF", 11: "VT", 8: "BS", 7: "BEL"}
    print("STRAY CONTROL CHARACTERS:",
          {names.get(b, b): n for b, n in control.items()})
    for lineno, line in enumerate(raw.split(b"\n"), 1):
        if any(0 <= b < 32 and b != 10 for b in line):
            print("  line %d: %r" % (lineno, line[:90]))
else:
    print("stray control characters: none")

# ---- citations ----
bibkeys = set(re.findall(r"@(?:\w+)\{([^,]+),", bib))
cited = set()
for m in re.findall(rx(r"@cite[pt]?(?:\[[^\]]*\])*\{([^}]+)\}"), tex):
    for k in m.split(","):
        cited.add(k.strip())
print("citations: %d cited, %d in bib" % (len(cited), len(bibkeys)))
print("  missing keys:", sorted(cited - bibkeys) or "none")
print("  uncited entries:", len(bibkeys - cited))

# ---- figures ----
figs = re.findall(rx(r"@includegraphics\[[^\]]*\]\{([^}]+)\}"), tex)
missing = [f for f in figs if not os.path.exists(os.path.join(FIGDIR, f))]
print("figures: %d referenced, missing: %s" % (len(figs), missing or "none"))

# ---- cross-references ----
labels = set(re.findall(rx(r"@label\{([^}]+)\}"), tex))
refs = set(re.findall(rx(r"@ref\{([^}]+)\}"), tex))
print("undefined refs:", sorted(refs - labels) or "none")
print("unused labels:", len(labels - refs))

# ---- environments ----
for env in ["table", "table*", "figure", "figure*", "tabular", "equation",
            "document", "itemize", "enumerate", "abstract"]:
    b = len(re.findall(rx(r"@begin\{") + re.escape(env) + r"\}", tex))
    e = len(re.findall(rx(r"@end\{") + re.escape(env) + r"\}", tex))
    if b != e:
        print("UNBALANCED %s: %d begin vs %d end" % (env, b, e))
print("brace balance:", tex.count("{") - tex.count("}"))

# ---- section lengths (body only) ----
body = tex.split(BS + "appendix")[0]
parts = re.split(rx(r"\n@section\{([^}]+)\}"), body)
print("\nBODY SECTION LENGTHS (words)")
for i in range(1, len(parts), 2):
    words = len(re.sub(rx(r"@[a-zA-Z]+") + r"|[{}]", " ", parts[i + 1]).split())
    print("  %6d  %s" % (words, parts[i]))

# ---- claim-language audit ----
print("\nCLAIM LANGUAGE")
for term in ["provably", "proves", "defect", "fatal", "clearly shows",
             "causes", "leads to", "drives"]:
    hits = len(re.findall(r"\b" + term + r"\b", tex, flags=re.I))
    if hits:
        print("  %-14s %d" % (term, hits))

# ---- exploratory demarcation, main text vs appendix ----
main_expl = len(re.findall(r"[Ee]xploratory", body))
appx_expl = len(re.findall(r"[Ee]xploratory", tex)) - main_expl
print("\nexploratory mentions: body=%d appendix=%d" % (main_expl, appx_expl))
print("DATA REQUIRED markers:", tex.count("[DATA REQUIRED"))

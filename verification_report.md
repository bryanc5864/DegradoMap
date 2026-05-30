# Bibliography Verification Report — DegradoMap paper.tex
## Verified: 2026-04-11 | Protocol: Triple-Pass | Tool: /bibliography-verifier

---

## Summary

| Metric | Count |
|--------|-------|
| References processed | 24 |
| Fully verified (no changes) | 21 |
| Corrections made | 3 |
| Unverifiable | 0 |

**All 24 references exist and are real papers. Zero hallucinated references detected.**

---

## Corrections Log

| # | Key | Field | Original | Corrected | Source |
|---|-----|-------|----------|-----------|--------|
| 3 | pettersson2019pbd | title | "PROTACs---past, present and future" | "{PROteolysis} {TArgeting} {Chimeras} ({PROTACs})---past, present and future" | PubMed PMID 31200855; DOI 10.1016/j.ddtec.2019.01.002 |
| 22 | lin2017focal | pages | `2980--2988` | `2999--3007` | DBLP conf/iccv/LinGGHD17; DOI 10.1109/ICCV.2017.324; IEEE Xplore |
| 24 | gligorijevic2021deepfri | author[1] | `Gligori\'{c}evi\'{c}` | `Gligorijevi\'{c}` | PubMed PMID 34039967; Nature Comms DOI 10.1038/s41467-021-23303-9 |

---

## Pass-by-Pass Detail

### Pass 1: Title & Existence Verification

All 24 papers confirmed to exist with correct titles. No hallucinated papers found.

Notable confirmations:
- lin2017focal: "Focal Loss for Dense Object Detection" — correct title confirmed via CVF/arXiv
- gligorijevic2021deepfri: title correct; first author name flagged for Pass 2 investigation

### Pass 2: Author Verification

Systematic positional author comparison for all 24 references.

**High author-count papers (verified fully):**
- li2022deepprotacs (12 authors): all confirmed correct — Fenglei Li ... Fang Bai
- zhang2022mapd (14 authors): all confirmed correct — Wubing Zhang ... X Shirley Liu
- jumper2021alphafold (34 total, first 3 cited): John Jumper, Richard Evans, Alexander Pritzel — confirmed
- tsherniak2017depmap (25 total, first 3 cited): Aviad Tsherniak, Francisca Vazquez, Phil G. Montgomery — confirmed
- li2017ubibrowser (13 total, 6 cited): Yang Li, Ping Xie, Liang Lu, Jian Wang, Lihong Diao, Zhongyang Liu — confirmed
- elnaggar2022prottrans (12 total, 3 cited): Ahmed Elnaggar, Michael Heinzinger, Christian Dallago — confirmed

**Correction found:**
- gligorijevic2021deepfri: First author's last name was `Gligorićević` (two ć-acutes) but correct name is `Gligorijević` (plain j + terminal ć). The LaTeX was `Gligori\'{c}evi\'{c}` (wrong) → corrected to `Gligorijevi\'{c}` (correct).

**Special cases confirmed correct:**
- schutt2017schnet: "H.E. Sauceda Felix" = Huziel Enoc Sauceda Felix (compound Spanish surname) ✓
- satorras2021egnn: "V.G. Satorras" = Víctor Garcia Satorras (compound Spanish surname, accent on í) ✓
- liu2025degrademaster: "M.J. Roy" = Michael J. Roy (confirmed second author) ✓

### Pass 3: Full Re-Verification + Venue/Pages

All venues, years, volumes, and page numbers re-verified via fresh searches.

**Corrections found:**
- pettersson2019pbd: Title was abbreviated. "PROTACs---past, present and future" is a shortened form; the published title includes the acronym expansion: "PROteolysis TArgeting Chimeras (PROTACs)---past, present and future". Updated.
- lin2017focal: ICCV 2017 pages `2980--2988` were wrong. Correct pages per DBLP and IEEE DOI are `2999--3007`. Some secondary databases (SCIRP) propagate the incorrect 2980-2988 range. Updated.

**Confirmed metadata (selected highlights):**
- schutt2017schnet: NeurIPS 2017 pages 991-1001 — confirmed from proceedings PDF metadata
- satorras2021egnn: PMLR v139 pages 9323-9332 — confirmed from PMLR metadata
- zengerle2015selective: ACS Chem Biol 10(8):1770-1777 — confirmed from PubMed PMID 26035625
- guo2017calibration: PMLR v70 pages 1321-1330 — confirmed
- lin2023esm2: Science 379(6637):1123-1130 — confirmed
- elnaggar2022prottrans: IEEE TPAMI 44(10):7112-7127 — confirmed; note published title uses "Toward" not "Towards"

---

## Notes on et al. Usage

Several references use "et al." in the LaTeX source. The visible named authors were verified in all cases:
- jumper2021alphafold: J. Jumper, R. Evans, A. Pritzel ✓ (34 total)
- lin2023esm2: Z. Lin, H. Akin, R. Rao ✓ (15 total)
- elnaggar2022prottrans: A. Elnaggar, M. Heinzinger, C. Dallago ✓ (12 total)
- tsherniak2017depmap: A. Tsherniak, F. Vazquez, P.G. Montgomery ✓ (25 total)
- li2017ubibrowser: Y. Li, P. Xie, L. Lu, J. Wang, L. Diao, Z. Liu ✓ (13 total)
- gligorijevic2021deepfri: V. Gligorijević, P.D. Renfrew, T. Kosciolek ✓ (14 total)

---

## Conclusion

The DegradoMap bibliography is clean. Three factual errors were found and corrected:
1. One author name misspelling (gligorijevic — missing 'j')
2. One page number error (focal loss — off by ~20 pages)
3. One abbreviated title (pettersson — missing acronym expansion)

No hallucinated papers, fabricated authors, or wrong venues were found.

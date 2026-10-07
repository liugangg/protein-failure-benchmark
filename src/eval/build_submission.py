"""生成 bioRxiv 投稿件 (DOCX + PDF + 纯文本摘要 + 字段清单)。

硬门: 先跑 check_en_sync.py —— **拒绝用陈旧的英文稿出投稿件**。
转换链: Markdown -> HTML (src/eval/md2html.py) -> LibreOffice -> DOCX / PDF。
  本机无 pandoc; 有 LibreOffice 与 pdflatex。选 LibreOffice 是因为它处理 25 张表格更稳,
  且 DOCX 可编辑 (作者块要人工补)。
转换后**机器核对**: 表格数与全部数值必须守恒, 不一致就报错 —— 自写转换器不能靠眼看。
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval.md2html import convert  # noqa: E402

EN = pathlib.Path("reports/default/PREPRINT_biorxiv_EN.md")
OUT = pathlib.Path("submission")
NUM = re.compile(r"\d[\d,]*\.?\d*")


def numset(t: str) -> set[str]:
    return {m.rstrip(".").replace(",", "") for m in NUM.findall(t)} - {""}


def docx_text(p: pathlib.Path) -> tuple[str, int]:
    with zipfile.ZipFile(p) as z:
        xml = z.read("word/document.xml").decode("utf8")
    ntbl = xml.count("<w:tbl>")
    txt = re.sub(r"<[^>]+>", " ", xml)
    return txt, ntbl


def main() -> None:
    r = subprocess.run([sys.executable, "src/eval/check_en_sync.py"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("!! 英文稿同步检查未通过, 拒绝生成投稿件:\n"
                         + r.stdout + r.stderr)
    print("英文稿同步检查: PASS")

    md = EN.read_text(encoding="utf8")
    title = md.splitlines()[0].lstrip("# ").strip()

    # ── DOI 与 TODO 的一致性门 (刘刚刚 2026-10-06: 预印本不该带着内部 TODO 上线) ──
    # configs/release.yaml 的 zenodo_doi 一旦填上, 稿件里就不许再出现占位符或待办句;
    # 反之 DOI 还没落实时, 占位符必须在 —— 不能悄悄变成一个看着像真的死 DOI。
    import yaml as _y
    _rel = _y.safe_load(pathlib.Path("configs/release.yaml").read_text()) or {}
    _doi = _rel.get("zenodo_doi")
    _ph = "to be added" in md
    _todo = "remains before posting" in md or "Delete this sentence before posting" in md
    if _doi:
        bad = []
        if _ph:
            bad.append("稿件里还有 \"to be added\" 占位符")
        if _todo:
            bad.append("稿件第 1 页还留着投稿前待办句")
        if f"doi:{_doi}" not in md and str(_doi) not in md:
            bad.append(f"稿件里找不到 {_doi}")
        if bad:
            raise SystemExit(
                "!! release.yaml 已填 zenodo_doi, 但稿件没同步, 拒绝出投稿件:\n  - "
                + "\n  - ".join(bad)
                + "\n   把 §7 / §9 的占位符换成实际 DOI, 并整句删掉第 1 页的待办。")
    else:
        if not _ph:
            raise SystemExit(
                "!! release.yaml 的 zenodo_doi 为空, 但稿件里没有占位符 —— "
                "是不是手写了一个 DOI? 未发布的 Zenodo DOI 不解析, 不能写进稿件。")
    print(f"DOI 门: zenodo_doi={_doi or '(未落实)'} · 占位符={'在' if _ph else '无'} "
          f"· 待办句={'在' if _todo else '无'}")
    OUT.mkdir(exist_ok=True)

    html_p = OUT / "manuscript.html"
    html_p.write_text(convert(md, title), encoding="utf8")

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise SystemExit("!! 找不到 LibreOffice")
    for fmt in ("docx:MS Word 2007 XML", "pdf"):
        subprocess.run([soffice, "--headless", "--convert-to", fmt,
                        "--outdir", str(OUT), str(html_p)],
                       check=True, capture_output=True, timeout=600)

    docx = OUT / "manuscript.docx"
    pdf = OUT / "manuscript.pdf"
    for f in (docx, pdf):
        if not f.exists():
            raise SystemExit(f"!! 转换没产出 {f}")

    # ── 机器核对: 自写转换器不能靠眼看 ──
    src_nums = numset(md)
    sep = re.compile(r"\|[-: |]+\|")
    n_tbl_md = len(sep.findall(md))

    dtxt, n_tbl_docx = docx_text(docx)
    lost_docx = sorted(src_nums - numset(dtxt))

    ptxt = ""
    if shutil.which("pdftotext"):
        subprocess.run(["pdftotext", str(pdf), str(OUT / "_p.txt")],
                       check=True, capture_output=True)
        ptxt = (OUT / "_p.txt").read_text(encoding="utf8", errors="replace")
        (OUT / "_p.txt").unlink()
    lost_pdf = sorted(src_nums - numset(ptxt)) if ptxt else []

    print(f"\nDOCX {docx.stat().st_size/1e6:.2f} MB  表格 {n_tbl_docx}/{n_tbl_md}  "
          f"丢失数值 {len(lost_docx)}")
    print(f"PDF  {pdf.stat().st_size/1e6:.2f} MB  丢失数值 "
          f"{len(lost_pdf) if ptxt else '(无 pdftotext, 未核)'}")

    # 结构核对: 列表**块**数守恒, 不是列表项数。
    # 第一版数的是项数, 结果毫无用处: 续行支持被破坏时每个标记行照样产生一个 <li>,
    # 项数不变 —— 实测 41/41 照过。真正的症状是**一个列表被拆成多个 <ol>**,
    # 于是编号从 1 重来 ("1. 1. 1.")。所以要比的是块数。
    html_txt = html_p.read_text(encoding="utf8")
    n_li_src = len(re.findall(r"^\s*(?:\d+\.|[-*])\s+\S", md, re.M))
    n_li_html = html_txt.count("<li>")

    def src_list_blocks(text: str) -> int:
        """源文件里连续的列表块数 (缩进续行不断开块)。"""
        blocks, inside = 0, False
        for ln in text.split("\n"):
            is_item = bool(re.match(r"^\s*(?:\d+\.|[-*])\s+\S", ln))
            is_cont = bool(ln.strip()) and ln[:1] in " \t" and not is_item
            if is_item and not inside:
                blocks += 1
                inside = True
            elif not is_item and not is_cont:
                inside = False
        return blocks

    n_blk_src = src_list_blocks(md)
    n_blk_html = html_txt.count("<ol>") + html_txt.count("<ul>")
    print(f"列表项 {n_li_html}/{n_li_src} · 列表块 {n_blk_html}/{n_blk_src}")

    problems = []
    if n_li_html != n_li_src:
        problems.append(f"列表项数 HTML {n_li_html} != 源 {n_li_src}")
    if n_blk_html != n_blk_src:
        problems.append(
            f"列表**块**数 HTML {n_blk_html} != 源 {n_blk_src} —— "
            f"有列表被拆开了, 编号会从 1 重来 (续行未被并入上一项)")
    if n_tbl_docx != n_tbl_md:
        problems.append(f"DOCX 表格数 {n_tbl_docx} != 源 {n_tbl_md}")
    if lost_docx:
        problems.append(f"DOCX 丢失数值 {lost_docx[:15]}")
    if ptxt and lost_pdf:
        problems.append(f"PDF 丢失数值 {lost_pdf[:15]}")
    if pdf.stat().st_size > 40e6:
        problems.append(f"PDF {pdf.stat().st_size/1e6:.1f} MB 超过 bioRxiv 的 40 MB 上限")
    if problems:
        raise SystemExit("!! 转换核对失败, 不要用这份投稿件:\n  - "
                         + "\n  - ".join(problems))

    # 纯文本摘要 (投稿表单要单独粘)
    m = re.search(r"^## Abstract\s*\n(.*?)(?=^## )", md, re.S | re.M)
    if not m:
        raise SystemExit("!! 抽不出 Abstract 小节")
    abst = m.group(1)
    abst = re.sub(r"\*\*|\*|`", "", abst)
    abst = re.sub(r"\n{2,}", "\n\n", abst).strip()
    (OUT / "abstract_plaintext.txt").write_text(abst + "\n", encoding="utf8")
    print(f"\nabstract_plaintext.txt  {len(abst):,} 字符 / "
          f"{len(abst.split()):,} 词")

    print(f"\n核对通过。产物在 {OUT}/")


if __name__ == "__main__":
    main()

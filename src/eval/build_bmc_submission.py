"""生成 BMC Bioinformatics 投稿件 (DOCX + PDF)。

硬门: 先跑 src/eval/check_bmc.py —— 不通过不出件。
转换链: Markdown -> HTML (src/eval/md2html.py) -> LibreOffice -> DOCX / PDF。
BMC 要求表格用文字处理软件的表格功能排版 (不得是图片或电子表格), 故 DOCX 是投稿正件,
PDF 仅供人眼复核。

转换后机器核对: 表格数与全部数值必须守恒。

**PDF 侧的换行连字符必须先归一化**: 长 URL 与页码区间会在行末断在连字符上
(例 `.../s12859-` 换行 `017-1995-z`), pdftotext 合行时把连字符吃掉, 于是
`12859` / `017` / `941` / `946` 看起来像"丢失"。这是抽取假象而非渲染缺陷
(DOCX 侧 0 丢失), 所以比对前消掉 `-\n`, 而不是把这些数字加进白名单 ——
白名单会把真正的丢失一起放过。

用法: .venv/bin/python src/eval/build_bmc_submission.py
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

MD = pathlib.Path("reports/default/BMC_manuscript_EN.md")
OUT = pathlib.Path("submission")
NUM = re.compile(r"\d[\d,]*\.?\d*")


def numset(t: str) -> set[str]:
    return {m.rstrip(".").replace(",", "") for m in NUM.findall(t)} - {""}


def dehyphenate(t: str) -> str:
    """消掉换行处的连字符断字, 使 `s12859-\n017` 还原为 `s12859-017`。

    必须配合 `pdftotext -layout`: 默认模式会自己合行并把连字符一并吃掉,
    届时已无 `-\n` 可归一化, 真假丢失再也分不开。
    """
    return re.sub(r"-[ \t]*\n[ \t]*", "-", t)


def main() -> None:
    r = subprocess.run([sys.executable, "src/eval/check_bmc.py"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("!! BMC 核验未通过, 拒绝出件:\n" + r.stdout + r.stderr)
    print("BMC 核验门: PASS")

    OUT.mkdir(exist_ok=True)
    md = MD.read_text(encoding="utf8")
    title = md.splitlines()[0].lstrip("# ").strip()

    html = OUT / "BMC_manuscript.html"
    html.write_text(convert(md, title), encoding="utf8")

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise SystemExit("!! 找不到 LibreOffice")
    for fmt in ("docx:MS Word 2007 XML", "pdf"):
        subprocess.run([soffice, "--headless", "--convert-to", fmt,
                        "--outdir", str(OUT), str(html)],
                       check=True, capture_output=True, timeout=900)

    docx, pdf = OUT / "BMC_manuscript.docx", OUT / "BMC_manuscript.pdf"
    for f in (docx, pdf):
        if not f.exists():
            raise SystemExit(f"!! 转换没产出 {f}")

    src_nums = numset(md)
    n_md_tbl = len(re.findall(r"\|[-: |]+\|", md))

    with zipfile.ZipFile(docx) as z:
        xml = z.read("word/document.xml").decode("utf8")
    n_docx_tbl = xml.count("<w:tbl>")
    lost_docx = sorted(src_nums - numset(re.sub(r"<[^>]+>", " ", xml)))

    if not shutil.which("pdftotext"):
        raise SystemExit("!! 找不到 pdftotext, 无法核验 PDF, 拒绝出件。")
    subprocess.run(["pdftotext", "-layout", str(pdf), str(OUT / "_bmc.txt")],
                   check=True, capture_output=True)
    ptxt = (OUT / "_bmc.txt").read_text(encoding="utf8", errors="replace")
    (OUT / "_bmc.txt").unlink()
    lost_pdf = sorted(src_nums - numset(dehyphenate(ptxt)))

    print(f"\nDOCX {docx.stat().st_size / 1e6:.2f} MB  表格 {n_docx_tbl}/{n_md_tbl}  "
          f"丢失数值 {len(lost_docx)}")
    print(f"PDF  {pdf.stat().st_size / 1e6:.2f} MB  丢失数值 {len(lost_pdf)}")

    problems = []
    if n_docx_tbl != n_md_tbl:
        problems.append(f"DOCX 表格数 {n_docx_tbl} != 源文件 {n_md_tbl}")
    if lost_docx:
        problems.append(f"DOCX 丢失数值 {lost_docx[:12]}")
    if lost_pdf:
        problems.append(f"PDF 丢失数值 {lost_pdf[:12]}")
    if "<http" in ptxt or "&lt;http" in ptxt:
        problems.append("PDF 里出现字面量 '<http' (裸链接写成了 <url>)")
    if pdf.stat().st_size > 40e6:
        problems.append(f"PDF {pdf.stat().st_size / 1e6:.1f} MB 超过 40 MB")
    if problems:
        raise SystemExit("!! 转换守恒核对未通过:\n  - " + "\n  - ".join(problems))

    print("链接门: PASS")
    print("\n核对通过。投稿正件 submission/BMC_manuscript.docx (PDF 仅供复核)")


if __name__ == "__main__":
    main()

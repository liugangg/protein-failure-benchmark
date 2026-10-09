"""打包 BMC Bioinformatics 投稿材料: 稿件正件 + cover letter + 清单。

本次投稿**只提交稿件正文**, 不提交图与 Additional file —— 稿件里也不得承诺它们
(src/eval/check_bmc.py 第 8 条门管这个)。

流程:
  1. 跑 src/eval/build_bmc_submission.py (内含 check_bmc.py 那道门) 出稿件 DOCX/PDF
  2. cover letter 的每个数值必须能在稿件里找到 —— 投稿信不许出现稿件里没有的数字
  3. cover letter 转 DOCX
  4. 落到 submission/BMC_submission_<日期>/ 并写 MANIFEST.txt (含 sha256)
  5. 打成一个 zip

用法: .venv/bin/python src/eval/package_bmc_submission.py
"""
from __future__ import annotations

import datetime
import hashlib
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval.md2html import convert  # noqa: E402

OUT = pathlib.Path("submission")
MD = pathlib.Path("reports/default/BMC_manuscript_EN.md")
CL = OUT / "COVER_LETTER.md"
NUM = re.compile(r"\d[\d,]*\.?\d*")

# cover letter 里允许出现、但稿件正文没有的数值: 地址、ORCID 分段、序号
CL_ALLOWED = {"0009", "0007", "6982", "201", "218", "199", "215123",
              "20", "1", "2", "3", "4", "5"}


def numset(t: str) -> set[str]:
    return {m.rstrip(".").replace(",", "") for m in NUM.findall(t)} - {""}


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    # 1) 稿件 —— 这一步内部已经跑过 check_bmc.py
    r = subprocess.run([sys.executable, "src/eval/build_bmc_submission.py"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("!! 稿件出件失败, 不打包:\n" + r.stdout + r.stderr)
    print(r.stdout.strip())

    if not CL.exists():
        raise SystemExit(f"!! 找不到 {CL}")

    # 2) cover letter 不许出现稿件里没有的数字
    orphan = sorted(numset(CL.read_text(encoding="utf8"))
                    - numset(MD.read_text(encoding="utf8")) - CL_ALLOWED)
    if orphan:
        raise SystemExit(f"!! cover letter 有稿件里找不到的数值: {orphan}\n"
                         "   投稿信的数字必须与稿件一致。")
    print("\ncover letter 数值门: PASS —— 无稿件外的数字")

    # 3) cover letter 转 DOCX
    cl_html = OUT / "_cover.html"
    cl_html.write_text(convert(CL.read_text(encoding="utf8"), "Cover letter"), encoding="utf8")
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise SystemExit("!! 找不到 LibreOffice")
    subprocess.run([soffice, "--headless", "--convert-to", "docx:MS Word 2007 XML",
                    "--outdir", str(OUT), str(cl_html)],
                   check=True, capture_output=True, timeout=600)
    cl_docx = OUT / "_cover.docx"
    if not cl_docx.exists():
        raise SystemExit("!! cover letter 转换没产出")
    final_cl = OUT / "COVER_LETTER.docx"
    cl_docx.replace(final_cl)
    cl_html.unlink()

    # 4) 打包目录
    stamp = datetime.date.today().strftime("%Y%m%d")
    pkg = OUT / f"BMC_submission_{stamp}"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)

    # 表格数现算, 不写死: 删掉补充文件那张表后从 15 变成 14, 写死就会脱钩
    n_tbl = len(re.findall(r"\|[-: |]+\|", MD.read_text(encoding="utf8")))

    items = [
        (OUT / "BMC_manuscript.docx", "01_Manuscript.docx",
         "投稿正件。标题页、摘要、关键词、Background、Methods、Results、Discussion、"
         f"Conclusions、缩写表、Declarations 八节、参考文献、{n_tbl} 个表格。"),
        (final_cl, "02_Cover_letter.docx", "投稿信。"),
        (OUT / "BMC_manuscript.pdf", "03_Manuscript_for_review_only.pdf",
         "仅供人眼复核的排版预览, 不是投稿件 —— BMC 要求表格为文字处理软件表格, 故正件是 DOCX。"),
    ]
    L = [f"BMC Bioinformatics 投稿材料  {stamp}",
         "稿件: Center-dependent generalization in public protein-failure data",
         "作者: Ganggang Liu", "",
         "本次投稿只提交稿件正文与投稿信: 无图, 无 Additional file,",
         "稿件内也未承诺任何 Additional file。支撑材料在公开仓库与已发布沉积中,",
         "见稿件 Availability of data and materials 一节。", "",
         "文件:"]
    for src, name, desc in items:
        if not src.exists():
            raise SystemExit(f"!! 缺 {src}")
        shutil.copy2(src, pkg / name)
        L.append(f"  {name}")
        L.append(f"      {src.stat().st_size:,} 字节")
        L.append(f"      sha256 {sha256(pkg / name)}")
        L.append(f"      {desc}")
    L.append("")
    L.append("提交到 BMC 系统时: 01 作为 Manuscript, 02 作为 Cover letter, 03 不要上传。")
    (pkg / "MANIFEST.txt").write_text("\n".join(L) + "\n", encoding="utf8")

    # 5) zip
    zp = OUT / f"BMC_submission_{stamp}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pkg.iterdir()):
            z.write(f, f"{pkg.name}/{f.name}")

    print(f"\n打包完成: {pkg}/")
    for f in sorted(pkg.iterdir()):
        print(f"  {f.stat().st_size:>9,}  {f.name}")
    print(f"\nzip: {zp}  ({zp.stat().st_size:,} 字节)")


if __name__ == "__main__":
    main()

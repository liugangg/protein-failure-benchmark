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
    _pending = bool(_rel.get("zenodo_deposit_pending"))
    _ph = "deposit not yet published" in md          # 明显占位符
    _v2 = "revised version of this preprint" in md   # "DOI 将在修订版补上"
    _todo = "remains before posting" in md or "Delete this sentence before posting" in md

    # 三态门。任何一态下稿件都不许带内部待办上线。
    if _todo:
        raise SystemExit("!! 稿件仍带投稿前内部待办句, 拒绝出件 —— 预印本不该带 TODO 上线。")
    if _pending and not _doi:
        bad = []
        if not _v2:
            bad.append('稿件未写明 "the deposit DOI will be added in a revised version"')
        if _ph:
            bad.append("稿件仍有未发布占位符, 与 pending 表述重复")
        # 沉积还没发布时, 任何"已归档"的**现在时**表述都是假陈述 ——
        # 读者照它去找会什么也找不到, 而这正是本文批评别人的那类问题。
        present_tense = [
            "are archived at Zenodo", "is archived at Zenodo",
            "archived at Zenodo;", "归档于 Zenodo,", "已归档于 Zenodo",
            "are deposited at Zenodo", "is deposited at Zenodo",
        ]
        hit = [x for x in present_tense if x in md]
        if hit:
            bad.append(f"稿件用现在时声称数据已归档, 但沉积尚未发布: {hit}")
        if bad:
            raise SystemExit("!! release.yaml 标为 zenodo_deposit_pending, 但稿件没同步:\n  - "
                             + "\n  - ".join(bad))
        print("DOI 门: 沉积待发布 —— 稿件已写明 DOI 将在修订版补上 (无占位符、无待办)")
    elif _doi:
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
                "!! zenodo_doi 为空且未标 pending, 但稿件里没有占位符 —— "
                "是不是手写了一个 DOI? 未发布的 Zenodo DOI 不解析, 不能写进稿件。")
        print("DOI 门: 未落实且未标 pending —— 稿件带明显占位符")
    # ── 仓库路径门 (刘刚刚 2026-10-07) ──
    # §7 有一句全称陈述: "all reports cited here are released in the companion repository",
    # §9 又列出一串仓库内路径。全称陈述必须机械可核 —— 稿件声称"在仓库中"的路径,
    # 必须真的被 git 跟踪, 一个都不能少。本文批评别人"记录说有、实际没有",
    # 自己不能犯同一条。
    #
    # NOT_SHIPPED 是稿件**明确声明不在仓库**的产物, 不受本门约束。
    # 但这个豁免名单自己也要受约束: 每一项都必须在稿件里找到那句声明,
    # 否则豁免就退化成"悄悄放过" —— 删掉 §7 那句话, 这里就会炸。
    NOT_SHIPPED = {
        "data/raw/": "redistributes no upstream raw file",
        "records.parquet": "deposited\nseparately rather than in the repository",
        "data/processed/records.parquet": "deposited\nseparately rather than in the repository",
        "data/interim/pooled_records.parquet": "under `data/interim/` are shipped as neither",
        "data/interim/pooled_unique_seqs.fasta": "under `data/interim/` are shipped as neither",
        "data/interim/split_groups.parquet": "under `data/interim/` are shipped as neither",
    }
    missing_decl = [k for k, v in NOT_SHIPPED.items() if v not in md]
    if missing_decl:
        raise SystemExit(
            "!! 以下产物被豁免'必须在仓库里'的检查, 但稿件里已找不到那句声明:\n  - "
            + "\n  - ".join(missing_decl)
            + "\n   要么把声明写回稿件, 要么把产物真的放进仓库 —— 不许静默豁免。")

    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout.split())
    if not tracked:
        raise SystemExit("!! git ls-files 返回空 —— 无法核验仓库路径声明, 拒绝出件。")

    cited = set(re.findall(
        r"`([A-Za-z0-9_][A-Za-z0-9_/.\-]*\.(?:md|json|yaml|parquet|py|txt|fasta))`", md))
    cited |= set(re.findall(r"`((?:src|configs|reports|data)/[A-Za-z0-9_/.\-]*/)`", md))

    def is_tracked(path: str) -> bool:
        if path in tracked:
            return True
        d = path.rstrip("/")
        return any(t.startswith(d + "/") for t in tracked)

    untracked = sorted(p for p in cited
                       if p not in NOT_SHIPPED and not is_tracked(p))
    if untracked:
        raise SystemExit(
            f"!! 稿件引用了 {len(untracked)} 个仓库内路径, 但它们没被 git 跟踪:\n  - "
            + "\n  - ".join(untracked)
            + "\n   §7 声称引用的报告全部随仓库发布 —— 这句话现在是假的。"
            + "\n   (若该产物本就不该进仓库, 在稿件里写明, 并登记进 NOT_SHIPPED。)")

    # §9.1 的指纹承诺: 表里每个产物都必须有一份被跟踪的 .prov.json。
    # 不用 `git ls-files '*.prov.json' | wc -l` 数个数 —— 仓库里另有中间产物与报告产物的
    # 指纹, 个数对不上, 而且"个数相等"从来不是完成判据; 这里按名字逐个点。
    fp_rows = re.findall(r"^\| `((?:data)/[^`]+)` \| [\d,]+ \| `[0-9a-f]{16}` \|$", md, re.M)
    if len(fp_rows) != 9:
        raise SystemExit(f"!! §9.1 指纹表解析到 {len(fp_rows)} 行, 预期 9 行 —— "
                         "表格式变了, 先修这个门再出件。")
    fp_missing = [f"{a}.prov.json" for a in fp_rows
                  if f"{a}.prov.json" not in tracked]
    if fp_missing:
        raise SystemExit(
            "!! §9.1 称这些指纹'随代码发布', 但它们没被 git 跟踪:\n  - "
            + "\n  - ".join(fp_missing)
            + "\n   注意 .gitignore 忽略了整个 data/interim/, 需要 git add -f。")
    print(f"仓库路径门: PASS —— 稿件引用 {len(cited) - len(NOT_SHIPPED)} 个仓库内路径全部被跟踪, "
          f"§9.1 的 9 份指纹全部在仓库 (另有 {len(NOT_SHIPPED)} 项已声明不在仓库)")

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

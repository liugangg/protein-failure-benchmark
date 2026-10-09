"""BMC Bioinformatics 版稿件的机械核验。

六项: (1) 正文与表格里每个数值都要能在冻结产物 (中文生成稿 + 已投英文稿) 的
数值集合里找到 —— 白名单只放引用编号、新 DOI 与 AUC 零假设值 0.5;
(2) 表格内不得有千位逗号 (BMC 规定); (3) 统一美式拼写, 参考文献标题照录原刊拼写;
(4) Vancouver 编号闭合且连续; (5) 不得残留中文编辑说明或旧稿 § 交叉引用;
(6) 有意留给作者的部分逐条列出, 不静默通过。

为什么要查 (1): 一处 0.703 被手打成 0.7030 就是这道门逮到的 —— 改动了打印精度。
用法: .venv/bin/python src/eval/check_bmc.py
"""
import re, pathlib, sys

BMC = pathlib.Path("reports/default/BMC_manuscript_EN.md")
md = BMC.read_text(encoding="utf8")
NUM = re.compile(r"\d[\d,]*\.?\d*")
def numset(t): return {m.rstrip(".").replace(",", "") for m in NUM.findall(t)} - {""}
fail = []

# ── 1) 数值可追溯性 ──
zh = pathlib.Path("reports/default/PREPRINT_biorxiv.md").read_text(encoding="utf8")
en = pathlib.Path("reports/default/PREPRINT_biorxiv_EN.md").read_text(encoding="utf8")
frozen = numset(zh) | numset(en)
# 排除: 参考文献(年/卷/页/DOI)、依赖版本行、补充文件表(编号)
body_lines = []
skip = False
for ln in md.split("\n"):
    if ln.startswith("## References"): skip = True
    if ln.startswith("## Tables"): skip = False
    if skip: continue
    if ln.startswith("Other requirements:"): continue
    if ln.startswith("| Additional file"): continue
    body_lines.append(ln)
body = "\n".join(body_lines)
NEW_OK = {"0.5",  # AUC 的零假设值, 定义常量而非测量值
          "23189683", "821654", "7844779", "15722219", "16", "17", "18", "19", "20", "21",
          "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
          "33", "3766", "64331", "64379", "2024", "2025", "2026", "2023", "2017"}
orphan = sorted(numset(body) - frozen - NEW_OK, key=lambda x: (len(x), x))
if orphan:
    fail.append(f"数值无法追溯到冻结产物 ({len(orphan)}): {orphan}")
else:
    print(f"✅ 数值可追溯: 正文/表格全部命中冻结集合 (白名单 {len(NEW_OK)} 项为引用编号与新 DOI)")

# ── 2) 表格内不得有千位逗号 (BMC 规定) ──
bad_rows = [ln.strip()[:95] for ln in md.split("\n")
            if ln.lstrip().startswith("|") and re.search(r"\d,\d{3}", ln)]
if bad_rows:
    fail.append(f"表格行含千位逗号 ({len(bad_rows)}): " + " ⏎ ".join(bad_rows[:4]))
else:
    print("✅ 表格内无千位逗号")

# ── 3) 美式拼写 ──
BRIT = ["unfavourable", "favour", "behaviour", "centre", "analysed", "licence",
        "totalling", "generalisation", "normalised", "modelling", "labelling",
        "characterised", "organisation"]
hits = []
for w in BRIT:
    for m in re.finditer(r"\b" + w, md, re.I):
        line = md[:m.start()].count("\n") + 1
        ctx = md.split("\n")[line-1].strip()[:90]
        # 参考文献标题照录原刊拼写, 豁免
        if ctx.startswith(("7. Oeschey", "20. Overath")) or "generalisation:" in ctx or "characterised binders" in ctx:
            continue
        hits.append(f"L{line} {w}: {ctx}")
if hits:
    fail.append(f"英式拼写 ({len(hits)}): " + " ⏎ ".join(hits[:5]))
else:
    print("✅ 统一美式拼写 (参考文献标题照录原刊拼写已豁免)")

# ── 4) Vancouver 编号闭合 ──
cited = set()
for m in re.finditer(r"\[(\d+(?:\s*[,-]\s*\d+)*)\]", md):
    for part in m.group(1).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); cited |= set(range(int(a), int(b) + 1))
        else: cited.add(int(part))
listed = {int(m.group(1)) for m in re.finditer(r"^(\d+)\. ", md, re.M)}
miss = sorted(cited - listed); unused = sorted(listed - cited)
if miss: fail.append(f"正文引用了但文献表没有: {miss}")
if unused: fail.append(f"文献表有但正文未引用: {unused}")
if not miss and not unused:
    print(f"✅ Vancouver 编号闭合: 引用 {len(cited)} 条 = 列出 {len(listed)} 条, 连续 1-{max(listed)}")

# ── 5) 不得残留编辑说明 / 旧稿 § 交叉引用 ──
res = []
if "编辑说明" in md or "搬运来源" in md: res.append("残留中文编辑说明")
sec = re.findall(r"§\s*\d", md)
if sec: res.append(f"残留旧稿 § 交叉引用 {len(sec)} 处")
if res: fail.append("; ".join(res))
else: print("✅ 无编辑说明残留、无旧稿 § 交叉引用")

# ── 7) 稿件引用的仓库路径必须真被 git 跟踪 ──
# 和 bioRxiv 版那道门同一条纪律: 声称"在仓库里"的东西必须真在仓库里。
import subprocess
tracked = set(subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split())
cited_paths = set(re.findall(r"`((?:src|configs|reports|data)/[A-Za-z0-9_/.\-]+\.[a-z]+)`", md))
cited_paths |= set(re.findall(r"`([A-Z][A-Z_]*\.md)`", md))
untracked = sorted(x for x in cited_paths if x not in tracked)
if untracked:
    fail.append(f"稿件引用的仓库路径未被 git 跟踪: {untracked}")
else:
    print(f"✅ 仓库路径: 引用的 {len(cited_paths)} 个路径全部被 git 跟踪")

# ── 8) 本次投稿只提交正文, 不得承诺任何 Additional file ──
# 承诺了却不提交, 正是本稿反复在清的那类陈述 (记录说有, 实际没有)。
promised = [ln.strip()[:100] for ln in md.split("\n")
            if re.search(r"(supplied|provided|included|attached)\s+as\s+an?\s+additional file", ln, re.I)
            or ln.lstrip().startswith("| Additional file")]
if promised:
    fail.append(f"稿件承诺了 Additional file 但本次只提交正文 ({len(promised)}): " + " / ".join(promised[:3]))
else:
    print("✅ 未承诺任何 Additional file (本次只提交稿件正文)")

# ── 6) 有意留空的部分, 明确列出而不是静默通过 ──
todo = [ln.strip() for ln in md.split("\n") if "TO BE WRITTEN" in ln]
print(f"\n⚠ 有意留给作者的部分: {len(todo)} 处")
for t in todo: print(f"    {t[:150]}")

print()
w = len(re.sub(r"[|#*\-]", " ", md).split())
print(f"全文约 {w} 词; 摘要 {len(' '.join(md.split('## Abstract')[1].split('## Keywords')[0].split()).split())} 词 (BMC 上限 350)")
if fail:
    print("\n!! 未通过:")
    for f in fail: print(f"  - {f}")
    sys.exit(1)
print("\n核验通过。")

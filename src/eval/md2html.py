"""最小 Markdown -> HTML 转换器, 只覆盖本项目稿件实际用到的语法。

为什么自己写而不装 pandoc: 本机没有 pandoc, 而全局纪律要求装任何东西前先读官方文档、
优先用现成办法。本机已有 LibreOffice (能把 HTML 转 DOCX/PDF), 所以缺的只是 MD->HTML
这一段, 且稿件语法很窄 (标题/粗体/行内代码/管道表格/引用块/列表/分隔线)。
自己写 60 行可控, 且**转换后有机器核对** (表格数与全部数值必须守恒), 不会静默丢内容。
"""
from __future__ import annotations

import html
import re


# 允许透传的行内标签: 只有上下标。署名块的单位编号需要真上标, 否则 PDF 里是
# "Ganggang Liu1,2,*" 这种看着像笔误的东西。白名单之外的 HTML 一律转义。
_ALLOW = ("sup", "sub")


# Markdown 反斜杠转义: \* \_ \` 等。必须在套用强调/代码正则**之前**挡下来,
# 否则 \* 要么被当成强调起点, 要么像 2026-10-06 那次一样把反斜杠原样印进 PDF
# ("Ganggang Liu1,2,\*")。做法是先换成私用区占位符, 格式化完再换回字面字符。
_ESC = re.compile(r"\\([*_`\[\]()#+\-.!\\])")
_PUA = 0xE000


def _protect(s: str) -> tuple[str, dict[str, str]]:
    table: dict[str, str] = {}

    def sub(m):
        ch = m.group(1)
        key = chr(_PUA + len(table))
        table[key] = ch
        return key
    return _ESC.sub(sub, s), table


def _inline(s: str) -> str:
    s, esc = _protect(s)
    s = html.escape(s, quote=False)
    for t in _ALLOW:
        s = s.replace(f"&lt;{t}&gt;", f"<{t}>").replace(f"&lt;/{t}&gt;", f"</{t}>")
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    # [text](url) -> <a>。稿中目前只有 ORCID 一处, 但不支持会直接把
    # "[0009-...](https://...)" 原样印进 PDF, 属于肉眼容易漏过的版式错误。
    s = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", s)
    for k, ch in esc.items():                      # 还原被保护的字面字符
        s = s.replace(k, html.escape(ch, quote=False))
    return s


def convert(md: str, title: str = "") -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)          # 去注释
    out: list[str] = []
    lines = md.split("\n")
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        st = ln.strip()

        if not st:
            i += 1
            continue

        if st.startswith("|") and i + 1 < n and re.fullmatch(
                r"\|[\s:|-]+\|", lines[i + 1].strip()):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], rows[2:]
            out.append("<table border='1' cellspacing='0' cellpadding='4'>")
            out.append("<thead><tr>" + "".join(
                f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>")
            for r in body:
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table>")
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", st)
        if m:
            lv = len(m.group(1))
            out.append(f"<h{lv}>{_inline(m.group(2))}</h{lv}>")
            i += 1
            continue

        if re.fullmatch(r"-{3,}", st):
            out.append("<hr/>")
            i += 1
            continue

        if st.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote><p>" + _inline(" ".join(buf)) + "</p></blockquote>")
            continue

        # 列表: 支持**续行** (缩进的后续行并入上一项)。
        # 不支持的后果不是报错而是静默错乱: 2026-10-06 实测, 带缩进续行的 3 项列表
        # 在 PDF 里渲染成 "1. 1. 1." —— 而数值守恒检查对这种结构错乱完全无感,
        # 所以 build_submission.py 另加了 <li> 计数核对。
        for marker, tag in ((r"\d+\.", "ol"), (r"[-*]", "ul")):
            if not re.match(rf"^{marker}\s+", st):
                continue
            items: list[str] = []
            while i < n:
                cur = lines[i]
                if re.match(rf"^\s*{marker}\s+", cur):
                    items.append(re.sub(rf"^\s*{marker}\s+", "", cur).strip())
                    i += 1
                elif items and cur.strip() and cur[:1] in " \t":
                    items[-1] += " " + cur.strip()       # 缩进续行 -> 并入上一项
                    i += 1
                else:
                    break
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for x in items)
                       + f"</{tag}>")
            break
        else:
            out.append(f"<p>{_inline(st)}</p>")
            i += 1
        continue

    css = ("body{font-family:'Liberation Serif',Georgia,serif;font-size:11pt;"
           "line-height:1.5;max-width:46em}"
           "table{border-collapse:collapse;font-size:9pt;margin:1em 0}"
           "th{background:#eee;text-align:left}"
           "code{font-family:'Liberation Mono',monospace;font-size:9.5pt}"
           "blockquote{margin:1em 0 1em 1.5em;color:#333}"
           "h1{font-size:17pt}h2{font-size:13pt}h3{font-size:11.5pt}")
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title><style>{css}</style></head>"
            f"<body>\n" + "\n".join(out) + "\n</body></html>")

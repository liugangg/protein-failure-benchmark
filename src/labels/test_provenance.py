"""provenance 自检 —— 三种事故各造一次, 确认都能被硬失败拦下。"""
import json, pathlib, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from labels.provenance import fingerprint, stamp, require

with tempfile.TemporaryDirectory() as td:
    d = pathlib.Path(td)
    up = d / "upstream.tsv"; up.write_text("a\tb\n1\t2\n")
    down = d / "downstream.tsv"; down.write_text("x\n")
    stamp(down, [up], produced_by="test")

    require([down])                                  # 一致 -> 通过
    print("  1/4 一致时通过")

    up.write_text("a\tb\n1\t2\n3\t4\n")              # 事故: 上游变了
    try:
        require([down]); raise AssertionError("上游变了却没拦下")
    except SystemExit as e:
        assert "上游" in str(e) and "已变" in str(e)
    print("  2/4 上游变更被拦下")

    stamp(down, [up], produced_by="test")            # 重新盖章后恢复
    require([down])
    down.write_text("x\ny\n")                        # 事故: 产物被改写
    try:
        require([down]); raise AssertionError("产物改写却没拦下")
    except SystemExit as e:
        assert "产物本身与记录不符" in str(e)
    print("  3/4 产物被改写被拦下")

    orphan = d / "orphan.parquet"; orphan.write_bytes(b"x")
    try:
        require([orphan]); raise AssertionError("缺指纹却没拦下")
    except SystemExit as e:
        assert "缺" in str(e)
    print("  4/4 缺指纹被拦下 (strict)")
    require([orphan], strict=False)                  # 显式放过

print("provenance self-check: 4/4 PASS")

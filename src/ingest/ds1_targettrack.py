"""DS1: PSI TargetTrack -> 统一六阶段标签记录。

走 TargetsbyContributor/<CENTER>.xml.gz 而不是整库 tt.xml.gz, 原因:
  1. center 直接来自文件名, 不用从 XML 里猜 —— GATE 1 的实验室分布和阶段二的
     实验室留出切分都靠这一列, 猜错代价极大。
  2. 41 个文件可以按文件并行, 这台机器 128 核。
  3. 单文件常驻内存小, 用 iterparse 逐 target 清理即可。

粒度: 一条记录 = 一个 trial (不是一个 target)。
理由: 同一个 target 常有多个 trial (不同截断/突变/标签构建), 它们的序列不同、
结局也不同。按 target 合并会把"这个构建失败了"和"另一个构建成功了"糊成一条。
但这带来重复序列问题 -> 交给 30% 聚类去冗余那一步处理 (SPEC §2.4 第 4 项),
而不是在这里提前合并。

输出: data/interim/ds1_targettrack/<CENTER>.parquet, 列见 labels.schema.RECORD_COLUMNS
      + 几列 DS1 专属诊断列 (conflict / raw_status / raw_stop_status / n_status_steps)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import pathlib
import sys
from collections import Counter

import polars as pl
import yaml
from lxml import etree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from labels.schema import (  # noqa: E402
    OBSERVED_FAIL,
    OBSERVED_SUCCESS,
    STAGE_INDEX,
    STAGES,
    UNOBSERVED,
    validate_label,
)

# 解压产物放 data/interim/extracted/ 而不是 data/raw/。
# 项目 CLAUDE.md: "data/raw/ — 原始下载, **只读**, 任何情况下不修改"。
# 解压出来的东西是派生数据, 可随时从 raw 里的归档重建, 不该占 raw。
TT_DIR = pathlib.Path(
    "data/interim/extracted/TargetTrack-1Jul2017/TargetsbyContributor"
)
OUT_DIR = pathlib.Path("data/interim/ds1_targettrack")
CONFIG = pathlib.Path("configs/targettrack_status_map.yaml")

AA = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


def _txt(el, tag: str) -> str:
    """取直接子元素文本, 没有就返回 ""。"""
    c = el.find(tag)
    if c is None or c.text is None:
        return ""
    return c.text.strip()


# 核酸字母表。**A/C/G/T 全是合法氨基酸字母**, 所以单靠"是否都在氨基酸表里"
# 无法区分 DNA 与蛋白 —— 这是 2026-09-30 抓到的严重 bug 的根因:
# 89,507 条 (DS1 的 9.46%) DNA 序列被当成蛋白入库, 且带着 express 失败 21,989 条等标签,
# 集中在 NYCOMPS / NYSGXRC / CESG / SGX, 正是那几个"长度中位 789-1275、首字母不是 M"的中心
# (DNA 以 ATG 开头, 长度是蛋白的 3 倍)。
NUCLEOTIDE = set("ACGTUN")
_MIN_DNA_LEN = 30          # 太短的序列即使全是 ACGT 也可能是真肽段, 不判为 DNA


def _looks_like_dna(s: str) -> bool:
    return len(s) >= _MIN_DNA_LEN and set(s) <= NUCLEOTIDE


def _clean_seq(raw: str) -> str:
    s = "".join(raw.split()).upper()
    if not s or not set(s) <= AA:
        return ""
    return "" if _looks_like_dna(s) else s


def _protein_seq_from_target(tsl) -> str:
    """从 targetSequenceList 里取**蛋白**序列。

    XML 结构 (实查 CESG.xml.gz): 每个 <targetSequence> 带 <sequenceChemicalType>,
    取值 dna / protein / rna, 且 **dna 那条排在 protein 前面**。
    所以必须按 sequenceChemicalType 选, 不能取第一个。
    """
    if tsl is None:
        return ""
    for ts in tsl.findall("targetSequence"):
        ct = _txt(ts, "sequenceChemicalType").lower()
        if ct and ct != "protein":
            continue
        olc = ts.find("oneLetterCode")
        if olc is not None and olc.text:
            s = _clean_seq(olc.text)
            if s:
                return s
    # 没有标注 protein 的, 退而取任何非 DNA 的
    for olc in tsl.findall(".//oneLetterCode"):
        if olc.text:
            s = _clean_seq(olc.text)
            if s:
                return s
    return ""


def _protein_seq_from_trial(tsl_trial) -> str:
    """从 trialSequenceList 里取蛋白序列。

    注意: <trialSequence> **没有** sequenceChemicalType 字段 (实查确认),
    只有裸的 <oneLetterCode>, 而且 DNA (id=1) 同样排在蛋白 (id=2) 前面。
    所以这里只能靠字母表判别 —— _clean_seq 已经把 DNA 挡掉了。
    """
    if tsl_trial is None:
        return ""
    for olc in tsl_trial.findall(".//oneLetterCode"):
        if olc.text:
            s = _clean_seq(olc.text)
            if s:
                return s
    return ""


class Mapper:
    def __init__(self, cfg: dict):
        self.reached = dict(cfg["status_to_reached_stage"])
        self.failed = dict(cfg["stop_status_to_failed_stage"])
        self.excl_status = set(cfg["exclude_records"]["trial_status_in"])
        self.excl_stop = set(cfg["exclude_records"]["stop_status_in"])
        inf = cfg.get("inferred_mappings", {}).get("structure_implies_stable", {})
        if not inf.get("enabled", True):
            # 关掉"解出结构 => stable" 这条推断: 把那批状态降级到 purify
            for s in inf.get("statuses", []):
                if self.reached.get(s) == "stable":
                    self.reached[s] = "purify"
        # SPEC v2 §2.4 的两条新规则
        ws = cfg.get("work_stopped_rule", {})
        self.ws_enabled = bool(ws.get("enabled", False))
        self.no_last_policy = cfg.get("no_last_state_policy", {}).get("policy", "drop")
        # 未在词表里出现过的状态要能被发现, 不能静默当 null
        self.unknown_status: Counter = Counter()
        self.unknown_stop: Counter = Counter()

    def reached_stage(self, status: str):
        if status not in self.reached:
            self.unknown_status[status] += 1
            return None
        return self.reached[status]

    def failed_stage(self, stop: str):
        if stop not in self.failed:
            self.unknown_stop[stop] += 1
            return None
        return self.failed[stop]


def parse_center(path: pathlib.Path, cfg: dict) -> tuple[list[dict], dict]:
    mapper = Mapper(cfg)
    center = path.name.replace(".xml.gz", "")
    rows: list[dict] = []
    stats = Counter()

    with gzip.open(path, "rb") as fh:
        # recover=True: 这批 2017 年的文件里有个别编码/实体问题, 不要因为一个
        # 坏 target 就丢掉整个 center。真丢了要在 stats 里看得见。
        ctx = etree.iterparse(
            fh, events=("end",), tag="target", recover=True, huge_tree=True
        )
        for _, target in ctx:
            try:
                stats["targets"] += 1
                target_id = target.get("id") or _txt(target, "targetId")
                date_created = _txt(target, "dateCreated")
                organism = ""
                designed = False

                tsl = target.find("targetSequenceList")
                if tsl is not None:
                    so = tsl.find(".//sourceOrganism/scientificName")
                    if so is not None and so.text:
                        organism = so.text.strip()
                for tpt in target.findall("targetProteinType"):
                    if tpt.text and "de novo" in tpt.text.lower():
                        designed = True
                for sct in tsl.findall(".//sequenceConstructType") if tsl is not None else []:
                    if sct.text and "de novo" in sct.text.lower():
                        designed = True

                tl = target.find("trialList")
                if tl is None:
                    stats["targets_no_trial"] += 1
                    continue

                for trial in tl.findall("trial"):
                    stats["trials"] += 1
                    trial_status = _txt(trial, "status")
                    # 不物理删除: 打标记保留, 由 pool.py 决定是否纳入汇总
                    exclusion_reason = ""
                    if trial_status in mapper.excl_status:
                        stats["excluded_test_target"] += 1
                        exclusion_reason = "test_target"

                    stop_status = ""
                    sd = trial.find("stopDetails")
                    if sd is not None:
                        stop_status = _txt(sd, "stopStatus")
                    if not exclusion_reason and stop_status in mapper.excl_stop:
                        stats["excluded_duplicate"] += 1
                        exclusion_reason = "duplicate_target"

                    # --- 序列: 优先 trialSequence (实际做的那条构建) ---
                    # 两条路径都必须挑**蛋白**, 不能取第一个 oneLetterCode:
                    # DNA 在 XML 里排在蛋白前面 (见 _protein_seq_from_* 的说明)。
                    seq = _protein_seq_from_trial(trial.find("trialSequenceList"))
                    if not seq:
                        seq = _protein_seq_from_target(tsl)
                    if not seq:
                        stats["trials_no_protein_sequence"] += 1
                        continue

                    # --- statusHistory: 成功走到哪 + 年份 ---
                    reached_idx = -1
                    dates: list[str] = []
                    n_steps = 0
                    shl = trial.find("statusHistoryList")
                    if shl is not None:
                        for sh in shl.findall("statusHistory"):
                            n_steps += 1
                            st = _txt(sh, "status")
                            dc = _txt(sh, "dateComplete")
                            if dc:
                                dates.append(dc)
                            stage = mapper.reached_stage(st)
                            if stage is not None:
                                reached_idx = max(reached_idx, STAGE_INDEX[stage])
                    # trial 的当前 status 也算一次证据
                    stage = mapper.reached_stage(trial_status) if trial_status else None
                    if stage is not None:
                        reached_idx = max(reached_idx, STAGE_INDEX[stage])

                    label = [UNOBSERVED] * len(STAGES)
                    for i in range(reached_idx + 1):
                        label[i] = OBSERVED_SUCCESS

                    # --- stopStatus: 失败在哪 + 与 statusHistory 的冲突检查 ---
                    conflict = False
                    evidence_tier = "none"
                    failed_stage = mapper.failed_stage(stop_status) if stop_status else None
                    if failed_stage == "EXCLUDE":       # 双保险, 上面已挡
                        exclusion_reason = exclusion_reason or "duplicate_target"
                        failed_stage = None
                    if exclusion_reason:
                        # 被排除的记录不产生任何标签, 但整条留着
                        label = [UNOBSERVED] * len(STAGES)
                        failed_stage = None
                    if failed_stage is not None:
                        fi = STAGE_INDEX[failed_stage]
                        if fi <= reached_idx:
                            # conflict_policy: status_history_wins
                            conflict = True
                            stats["conflict_stop_before_reached"] += 1
                        else:
                            label[fi] = OBSERVED_FAIL
                            for j in range(fi + 1, len(STAGES)):
                                label[j] = UNOBSERVED
                            evidence_tier = "explicit"
                            stats["neg_explicit"] += 1

                    # --- SPEC v2 §2.4: work stopped -> 已达到最后状态的下一步标 0 ---
                    # 只在 stopStatus 没能给出失败阶段时套用 (显式记录优先于推断)。
                    is_ws = (trial_status == "work stopped")
                    if mapper.ws_enabled and is_ws and evidence_tier == "none" and not exclusion_reason:
                        if reached_idx < 0:
                            # "连最后状态都取不到" —— 见 configs 的 no_last_state_policy
                            if mapper.no_last_policy == "keep_unobserved":
                                # 保留记录, 六阶段全 -1, 不产生阴性 (刘刚刚 2026-09-30 定)
                                stats["ws_no_last_state_kept"] += 1
                                exclusion_reason = "no_last_state_work_stopped"
                                label = [UNOBSERVED] * len(STAGES)
                            elif mapper.no_last_policy == "drop":
                                stats["ws_dropped_no_last_state"] += 1
                                continue
                            elif mapper.no_last_policy == "clone_zero":
                                label[STAGE_INDEX["clone"]] = OBSERVED_FAIL
                                for j in range(STAGE_INDEX["clone"] + 1, len(STAGES)):
                                    label[j] = UNOBSERVED
                                evidence_tier = "inferred_next_step"
                                stats["neg_inferred"] += 1
                                stats["ws_clone_zero_from_selected"] += 1
                            else:
                                raise ValueError(
                                    f"no_last_state_policy 取值非法: {mapper.no_last_policy}"
                                )
                        elif reached_idx < len(STAGES) - 1:
                            ni = reached_idx + 1
                            label[ni] = OBSERVED_FAIL
                            for j in range(ni + 1, len(STAGES)):
                                label[j] = UNOBSERVED
                            evidence_tier = "inferred_next_step"
                            stats["neg_inferred"] += 1
                        else:
                            # 已走到 bind (最后一个阶段) 还 work stopped -> 没有"下一步"
                            stats["ws_at_last_stage"] += 1

                    validate_label(label)

                    year = -1
                    all_dates = sorted(d for d in dates if len(d) >= 4 and d[:4].isdigit())
                    if all_dates:
                        year = int(all_dates[-1][:4])       # 最后一次状态变更的年份
                    elif len(date_created) >= 4 and date_created[:4].isdigit():
                        year = int(date_created[:4])

                    rows.append(
                        {
                            "record_id": f"ds1:{center}:{target_id}:t{trial.get('id')}",
                            "source": "ds1_targettrack",
                            "sequence": seq,
                            "seq_len": len(seq),
                            "target_id": str(target_id),
                            "center": center,
                            "year": year,
                            "organism": organism,
                            "is_designed": designed,
                            **{f"label_{s}": v for s, v in zip(STAGES, label)},
                            "evidence": f"status={trial_status}|stop={stop_status}",
                            "evidence_tier": evidence_tier,
                            "exclusion_reason": exclusion_reason,
                            "conflict": conflict,
                            "raw_status": trial_status,
                            "raw_stop_status": stop_status,
                            "n_status_steps": n_steps,
                            "year_first": int(all_dates[0][:4]) if all_dates else -1,
                        }
                    )
            finally:
                # iterparse 内存管理: 清掉已处理的 target 和它的前驱兄弟
                target.clear()
                while target.getprevious() is not None:
                    del target.getparent()[0]

    diag = {
        "center": center,
        "records": len(rows),
        **{k: int(v) for k, v in stats.items()},
        "unknown_status_values": dict(mapper.unknown_status),
        "unknown_stop_status_values": dict(mapper.unknown_stop),
    }
    return rows, diag


def _worker(args):
    path, cfg = args
    try:
        rows, diag = parse_center(path, cfg)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        if rows:
            pl.DataFrame(rows).write_parquet(OUT_DIR / f"{diag['center']}.parquet")
        return diag
    except Exception as e:  # 单个 center 失败不要拖垮全部, 但必须显式报出来
        return {"center": path.name, "ERROR": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", nargs="*", help="只跑指定 center, 调试用")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    files = sorted(TT_DIR.glob("*.xml.gz"))
    files = [f for f in files if not f.name.startswith("._")]
    if args.only:
        files = [f for f in files if f.name.replace(".xml.gz", "") in args.only]
    if not files:
        raise SystemExit(f"{TT_DIR} 下没有 center xml, 先解压 tarball")

    print(f"parsing {len(files)} center files with {args.workers} workers")
    diags = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for diag in ex.map(_worker, [(f, cfg) for f in files]):
            diags.append(diag)
            if "ERROR" in diag:
                print(f"  !! {diag['center']}: {diag['ERROR']}")
            else:
                print(
                    f"  {diag['center']:14s} rec={diag['records']:7d} "
                    f"expl={diag.get('neg_explicit',0):6d} "
                    f"infer={diag.get('neg_inferred',0):6d} "
                    f"ws_drop={diag.get('ws_dropped_no_last_state',0):6d} "
                    f"conflict={diag.get('conflict_stop_before_reached',0):6d} "
                    f"excl_dup={diag.get('excluded_duplicate',0):5d}"
                )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "_parse_diagnostics.json").write_text(
        json.dumps(diags, indent=2, ensure_ascii=False)
    )

    total_unknown_status: Counter = Counter()
    total_unknown_stop: Counter = Counter()
    for d in diags:
        total_unknown_status.update(d.get("unknown_status_values", {}))
        total_unknown_stop.update(d.get("unknown_stop_status_values", {}))
    if total_unknown_status or total_unknown_stop:
        print("\n!! 词表外的值出现了, 说明映射表不完整, 必须补进 YAML 再重跑:")
        print("   status:", dict(total_unknown_status))
        print("   stopStatus:", dict(total_unknown_stop))
    else:
        print("\n所有 status / stopStatus 值都在官方词表映射内")

    errs = [d for d in diags if "ERROR" in d]
    print(f"\ncenters ok={len(diags)-len(errs)} failed={len(errs)}")
    agg = Counter()
    for d in diags:
        for k in ("records", "trials", "neg_explicit", "neg_inferred",
                  "ws_no_last_state_kept",
                  "ws_dropped_no_last_state", "ws_clone_zero_from_selected",
                  "ws_at_last_stage", "conflict_stop_before_reached",
                  "excluded_duplicate", "excluded_test_target",
                  "trials_no_protein_sequence"):
            agg[k] += d.get(k, 0)
    print(f"total records={agg['records']:,} / trials={agg['trials']:,}")
    print(f"  阴性证据等级: explicit(stopStatus 直接给出)={agg['neg_explicit']:,} "
          f"inferred_next_step(v2 §2.4 推断)={agg['neg_inferred']:,}")
    print(f"  v2 §2.4 那批 (work stopped 但取不到六阶段状态): "
          f"保留为全未观测={agg['ws_no_last_state_kept']:,} "
          f"物理丢弃={agg['ws_dropped_no_last_state']:,}")
    print(f"  work stopped 已在最后阶段无下一步={agg['ws_at_last_stage']:,}")
    print(f"  打排除标记但保留在 parquet 里: test target={agg['excluded_test_target']:,} "
          f"重复靶点={agg['excluded_duplicate']:,} (由 pool.py 过滤)")
    print(f"  stopStatus 与状态历史矛盾={agg['conflict_stop_before_reached']:,}")
    print(f"  取不到蛋白序列而跳过={agg['trials_no_protein_sequence']:,} "
          f"(含 DNA-only 的 trial; A/C/G/T 全是合法氨基酸字母, 必须按字母表与 "
          f"sequenceChemicalType 双重判别)")


if __name__ == "__main__":
    main()

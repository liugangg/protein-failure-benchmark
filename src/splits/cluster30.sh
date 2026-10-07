#!/usr/bin/env bash
# 30% 序列相似度聚类 —— SPEC §2.4 第 4 项 (有效样本量) 与 §3.1 (按簇整体切分) 的基础。
#
# 参数依据 (MMseqs2 官方 wiki "Clustering criteria" 一节, 不是抄博客):
#   --min-seq-id 0.3    30% 相似度门槛
#   --alignment-mode 3  让 min-seq-id 用**真正的序列一致度**(相同残基/比对列数),
#                       而不是默认那个"等效相似性得分"近似值。写论文要经得起问。
#   -c 0.8 --cov-mode 1 覆盖度按较短(非中心)序列算, 80% —— TargetTrack 里大量
#                       trial 是同一个 target 的截断/域片段, 用 cov-mode 1 才能
#                       把它们归到一起; 用默认 cov-mode 0 会把截断体判成新序列,
#                       有效样本量虚高、训练测试之间留下泄漏通道。
#   --cluster-reassign  wiki 明确指出级联聚类会遗留不再满足判据的成员, 用它修正。
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MMSEQS="$ROOT/.tools/mmseqs/bin/mmseqs"
IN="$ROOT/data/interim/pooled_unique_seqs.fasta"
OUTDIR="$ROOT/data/interim/cluster30"
TMP="${TMPDIR:-/tmp}/mmseqs_c30_$$"
THREADS="${THREADS:-64}"

mkdir -p "$OUTDIR" "$TMP"
"$MMSEQS" easy-cluster "$IN" "$OUTDIR/c30" "$TMP" \
    --min-seq-id 0.3 -c 0.8 --cov-mode 1 \
    --alignment-mode 3 --cluster-reassign \
    --threads "$THREADS" -v 2
rm -rf "$TMP"
wc -l "$OUTDIR/c30_cluster.tsv"

# 盖章: 记录本产物的上游 (pooled_unique_seqs.fasta), 供下游 provenance.require() 校验。
# 见 src/labels/provenance.py —— mmseqs 崩掉后读到残留 .m8 曾建出一套错误分组。
"$ROOT/.venv/bin/python" -c "
import sys, pathlib
sys.path.insert(0, '$ROOT/src')
from labels import provenance
provenance.stamp('$OUTDIR/c30_cluster.tsv', ['$IN'], produced_by='src/splits/cluster30.sh')
provenance.stamp('$OUTDIR/c30_rep_seq.fasta', ['$IN'], produced_by='src/splits/cluster30.sh')
print('[cluster30] 已盖指纹章')
"

#!/usr/bin/env bash
# =============================================================================
# nohup_run.sh —— 通用后台任务运行器 (python / bash 都支持)
#
#   * 日志统一放在 nohup_logs/ ; 用 setsid+nohup 完全脱离终端, 断线/关窗口都不受影响
#   * 可随时切换用 python 或 bash 运行
#
# 常见用法:
#   1) 改下面【默认任务】后直接跑:     bash nohup_run.sh
#   2) 临时指定任意命令:
#        bash nohup_run.sh run <名字> python <脚本.py> [参数...]
#        bash nohup_run.sh run <名字> bash   <脚本.sh> [参数...]
#   3) 管理(对 <名字>):
#        bash nohup_run.sh status  [名字]     # 看运行状态 (不给名字=列出全部)
#        bash nohup_run.sh tail    <名字>     # 实时看进度 (Ctrl-C 只退出查看, 不影响任务)
#        bash nohup_run.sh stop    <名字>
#        bash nohup_run.sh restart <名字>     # 用上次相同参数重启
#        bash nohup_run.sh list
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
LOGDIR="$REPO_ROOT/nohup_logs"
mkdir -p "$LOGDIR"

# 用哪个 python (默认 mimigenrec 环境, 可用环境变量覆盖)
PYTHON_BIN="${PYTHON_BIN:-/data/sht/conda/mimigenrec/bin/python}"
export PYTHONUNBUFFERED=1

# ==================== 【默认任务】按需修改这里 ====================
RUN_TYPE="python"                    # python | bash
SCRIPT="data/amazon18_download.py"   # 要运行的脚本
NAME="amazon18_dl"                   # 任务名 (决定日志/PID 文件名)
ARGS=( --config data/amazon18.conf )
# =================================================================

is_running() {
  local pf="$LOGDIR/$1.pid" sf="$LOGDIR/$1.sid"
  [ -f "$pf" ] || return 1
  local pid; pid="$(cat "$pf" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
  # 若记录了会话 id, 校验 pid 仍属于该会话, 防止 pid 被复用导致误判/误杀
  if [ -f "$sf" ]; then
    local sid cur
    sid="$(cat "$sf" 2>/dev/null)"
    cur="$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')"
    if [ -n "$sid" ] && [ -n "$cur" ] && [ "$cur" != "$sid" ]; then return 1; fi
  fi
  return 0
}

start_job() {
  local name="$1"; shift
  local type="$1"; shift
  local script="$1"; shift
  local log="$LOGDIR/$name.log" pidfile="$LOGDIR/$name.pid"

  if is_running "$name"; then
    echo "[$name] 已在运行 (pid=$(cat "$pidfile")) — 如需重启: bash nohup_run.sh restart $name"
    return 1
  fi

  local -a runner
  case "$type" in
    python) runner=("$PYTHON_BIN" -u "$script") ;;
    bash)   runner=(bash "$script") ;;
    *) echo "未知运行类型: $type (应为 python 或 bash)"; return 1 ;;
  esac

  cd "$REPO_ROOT"
  echo "启动 [$name] : $type $script $*"
  echo "日志 -> $log (追加写入, 不覆盖历史)"
  # 追加写入 + 每次运行的起始分隔, 避免历史日志被覆盖
  {
    echo ""
    echo "=================================================================="
    echo "[$(date '+%F %T')] START [$name] : $type $script $*"
    echo "=================================================================="
  } >> "$log"
  # setsid: 新建会话/进程组, 便于整棵树干净停止; nohup: 免疫挂断
  setsid nohup "${runner[@]}" "$@" >>"$log" 2>&1 &
  local jobpid=$!
  echo "$jobpid" > "$pidfile"
  # 记录会话 id (setsid 后 sid=pid); 停止时只结束该会话, 避免误伤其它任务
  sleep 0.2
  local sid; sid="$(ps -o sid= -p "$jobpid" 2>/dev/null | tr -d ' ')"
  echo "${sid:-$jobpid}" > "$LOGDIR/$name.sid"

  # 记录参数以便 restart
  printf '%s\n%s\n' "$type" "$script" > "$LOGDIR/$name.meta"
  if [ $# -gt 0 ]; then printf '%s\0' "$@" > "$LOGDIR/$name.args"; else : > "$LOGDIR/$name.args"; fi

  echo "PID -> $(cat "$pidfile")"
  echo "实时进度: bash nohup_run.sh tail $name"
}

stop_job() {
  local name="$1"
  local pf="$LOGDIR/$name.pid" sf="$LOGDIR/$name.sid"
  if ! is_running "$name"; then
    echo "[$name] 未在运行"
    rm -f "$pf" "$sf"
    return 0
  fi
  local pid; pid="$(cat "$pf")"
  # 优先用启动时记录的会话 id; 兼容旧任务则现场查询
  local sid; sid="$(cat "$sf" 2>/dev/null)"
  [ -n "$sid" ] || sid="$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')"
  local my_sid; my_sid="$(ps -o sid= -p $$ 2>/dev/null | tr -d ' ')"

  if [ -z "$sid" ] || [ "$sid" = "$my_sid" ] || [ "$sid" = "1" ]; then
    # 会话信息异常或与本脚本同会话: 只杀该 pid, 绝不按会话批量杀
    echo "[$name] 会话信息异常(sid=${sid:-空})，仅结束 pid=$pid"
    kill -TERM "$pid" 2>/dev/null
  else
    local targets
    targets="$(ps -eo pid=,sid= | awk -v s="$sid" '$2==s{print $1}')"
    if [ -n "$targets" ]; then kill -TERM $targets 2>/dev/null; else kill -TERM "$pid" 2>/dev/null; fi
  fi

  for _ in 1 2 3 4 5 6 7 8 9 10; do is_running "$name" || break; sleep 1; done
  if is_running "$name"; then
    local targets
    targets="$(ps -eo pid=,sid= | awk -v s="$sid" '$2==s{print $1}')"
    if [ -n "$targets" ]; then kill -KILL $targets 2>/dev/null; else kill -KILL "$pid" 2>/dev/null; fi
  fi
  rm -f "$pf" "$sf"
  echo "[$name] 已停止"
}

status_job() {
  if is_running "$1"; then echo "[$1] 运行中 (pid=$(cat "$LOGDIR/$1.pid"))"; else echo "[$1] 未运行"; fi
}

list_jobs() {
  echo "=== 任务列表 ($LOGDIR) ==="
  shopt -s nullglob
  local any=0
  for lf in "$LOGDIR"/*.log; do
    any=1
    local name; name="$(basename "$lf" .log)"
    if is_running "$name"; then st="运行中 (pid=$(cat "$LOGDIR/$name.pid"))"; else st="停止"; fi
    printf "  %-16s %s\n" "$name" "$st"
  done
  shopt -u nullglob
  [ "$any" = "1" ] || echo "  (暂无任务)"
}

restart_job() {
  local name="$1"
  stop_job "$name"
  local meta="$LOGDIR/$name.meta" args="$LOGDIR/$name.args"
  [ -f "$meta" ] || { echo "找不到 [$name] 的运行记录, 无法 restart (请用 run 重新指定)"; return 1; }
  local type script
  type="$(sed -n 1p "$meta")"; script="$(sed -n 2p "$meta")"
  local -a a=()
  if [ -f "$args" ]; then while IFS= read -r -d '' x; do a+=("$x"); done < "$args"; fi
  start_job "$name" "$type" "$script" "${a[@]}"
}

# ------------------------------ 分发 ------------------------------
cmd="${1:-}"
case "$cmd" in
  "")        start_job "$NAME" "$RUN_TYPE" "$SCRIPT" "${ARGS[@]}" ;;
  run)       shift
             [ $# -ge 3 ] || { echo "用法: bash nohup_run.sh run <名字> <python|bash> <脚本> [参数...]"; exit 1; }
             start_job "$@" ;;
  status)    shift; if [ -n "${1:-}" ]; then status_job "$1"; else list_jobs; fi ;;
  list)      list_jobs ;;
  tail)      shift; [ -n "${1:-}" ] || { echo "用法: bash nohup_run.sh tail <名字>"; exit 1; }
             tail -f "$LOGDIR/$1.log" ;;
  stop)      shift; [ -n "${1:-}" ] || { echo "用法: bash nohup_run.sh stop <名字>"; exit 1; }
             stop_job "$1" ;;
  restart)   shift; [ -n "${1:-}" ] || { echo "用法: bash nohup_run.sh restart <名字>"; exit 1; }
             restart_job "$1" ;;
  -h|--help|help) sed -n '2,22p' "$0" ;;
  *)         echo "未知命令: $cmd"; echo "用法: bash nohup_run.sh [run|status|list|tail|stop|restart] ..."; exit 1 ;;
esac

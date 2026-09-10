"""
单日精确案例回测 (Case Study)
===============================
用户需求: "更精确些进行测试, 比如 2026-08-04 当天提取出个股, 然后去比较 5 日内的交易情况。"

做法:
  1. 选一个交易日 D (默认 2026-08-04)
  2. 股票池: 默认按成交额 Top-500 (对齐此前全市场广度验证)
  3. 用与月度回测完全相同的规则在 D 日盘中扫描出全部命中信号股
     (近5日累计下跌 + 当根15min放量×2 + 15min涨幅0~2% + 今/昨量比 + 排除三根bar)
  4. 按用户规则: 取信号根 15min 绝对成交量最大的前 N 只 (默认5) 作为实际买入
  5. 对选出的每只: 逐日跟踪 T+1 ~ T+5 的收盘价/累计涨跌, 标注实际卖出点
     (日K close >= 买入价×1.01 即当日收盘卖出; 5日内未触发则第5日强平)
  6. 输出: CSV 明细 + 详细 HTML 报告 (wide 逐日对照表 + 收益轨迹)

数据源/缓存 key 与 screener_v2.py 月度回测保持一致 (daily_*, 15min_bs_*),
这样能最大程度复用 `.cache_v2` 已有数据。
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import screener_v2 as S  # noqa: E402  复用所有数据源/规则/缓存

OUT = Path(__file__).parent / 'results_case'


# ============ 逐日模拟 (T+1..T+hold_days) ============
def simulate_one_symbol(code: str, name: str, target_dt: date,
                        df15_valid: pd.DataFrame, ddf: pd.DataFrame,
                        trade_dates: list[date],
                        min_decline_pct: float = S.MIN_DECLINE_PCT,
                        hold_days: int = S.HOLD_DAYS) -> tuple[dict | None, list[dict]]:
    """
    与 S.backtest_one_symbol 相同的信号判定/买入逻辑, 但额外返回
    未来 hold_days 个交易日每一天的逐日状态 (收盘/累计涨跌/动作), 便于做精确对照.
    """
    decline_ok, decline_diag = S.check_recent_decline(ddf, target_dt, min_pct=min_decline_pct)
    if not decline_ok:
        return None, []

    today_valid = df15_valid[df15_valid['day'].dt.date == target_dt].reset_index(drop=True)
    hist_valid = df15_valid[df15_valid['day'].dt.date < target_dt]
    if today_valid.empty:
        return None, []
    sig = S.find_signal_bar(pd.concat([hist_valid.tail(60), today_valid], ignore_index=True))
    if sig is None:
        return None, []

    buy_price = float(sig['close'])
    if buy_price <= 0:
        return None, []
    shares = int(S.CAPITAL_PER_STOCK / buy_price / S.LOT) * S.LOT
    if shares <= 0:
        return None, []
    actual_capital = shares * buy_price
    target_price = buy_price * (1 + S.TP_PCT / 100.0)

    future_dates = [d for d in trade_dates if d > target_dt][:hold_days]
    if not future_dates:
        return None, []

    ddf_idx = ddf.copy()
    ddf_idx['_date'] = pd.to_datetime(ddf['日期']).dt.date
    row_by_date = {r['_date']: r for _, r in ddf_idx.iterrows()}

    track = []            # 逐日跟踪
    exit_date, exit_price, exit_reason, exit_offset = None, None, None, None

    for offset, d in enumerate(future_dates, start=1):
        row = row_by_date.get(d)
        rec = {
            'code': code, 'name': name, 'buy_date': target_dt.strftime('%Y-%m-%d'),
            'day_no': offset, 'date': d.strftime('%Y-%m-%d'),
            'buy_price': round(buy_price, 3),
            'day_close': None, 'day_cum_pct': None, 'action': '持有',
        }
        if row is None or pd.isna(row.get('收盘')):
            rec['action'] = '无日K(疑似停牌)'
            track.append(rec)
            continue
        close_p = float(row['收盘'])
        cum_pct = (close_p - buy_price) / buy_price * 100.0
        rec['day_close'] = round(close_p, 3)
        rec['day_cum_pct'] = round(cum_pct, 2)

        if exit_date is None:                       # 尚未离场
            if close_p >= target_price:
                rec['action'] = f'T+{offset}止盈卖出'
                exit_date, exit_price, exit_reason, exit_offset = d, close_p, f'T+{offset}止盈(>=1%)', offset
            elif offset == hold_days:
                rec['action'] = f'{hold_days}日未触发-强平'
                exit_date, exit_price, exit_reason, exit_offset = d, close_p, f'{hold_days}日未触发-强平', hold_days
            else:
                rec['action'] = '持有中'
        else:
            rec['action'] = '已离场(观察)'
        track.append(rec)

    if exit_date is None:                            # 兜底: 后几日都停牌 -> 取最近有效日
        for rec in reversed(track):
            if rec['day_close'] is not None and rec['action'] in ('持有中', '无日K(疑似停牌)'):
                exit_date = datetime.strptime(rec['date'], '%Y-%m-%d').date()
                exit_price = rec['day_close']
                exit_reason = f'{hold_days}日未触发-强平'
                exit_offset = rec['day_no']
                break
    if exit_date is None:
        return None, []

    pnl = (exit_price - buy_price) * shares
    pnl_pct = (exit_price - buy_price) / buy_price * 100.0
    result = {
        'code': code, 'name': name,
        'date': target_dt.strftime('%Y-%m-%d'),
        'signal_time': str(sig['signal_time']),
        'bar_change_pct': round(sig['change_pct'], 2),
        'bar_vol_ratio': round(sig['vol_ratio'], 2),
        'signal_bar_vol': int(sig.get('signal_bar_vol', 0)),
        'prev_bar_vol': int(sig.get('prev_bar_vol', 0)),
        'today_yesterday_vol_ratio': round(sig['ytd_vol_ratio'], 2),
        'decline_window': decline_diag['window'] if decline_diag else None,
        'decline_cum_pct': round(decline_diag['cum_pct'], 2) if decline_diag else None,
        'buy_price': round(buy_price, 3),
        'target_price': round(target_price, 3),
        'exit_date': exit_date.strftime('%Y-%m-%d') if exit_date else '-',
        'exit_price': round(exit_price, 3),
        'hold_days': exit_offset,
        'shares': shares,
        'capital_used': round(actual_capital, 2),
        'pnl': round(pnl, 2),
        'pnl_pct': round(pnl_pct, 2),
        'exit_reason': exit_reason,
        'selected': 0,          # 1 = 按量排序入选 Top-N
        'day_rank': 0,
    }
    return result, track


# ============ 主流程 ============
def run_case(target_date: str, sample_size: int, top: bool, max_positions: int,
             hold_days: int, max_workers: int, min_decline_pct: float) -> dict:
    S.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    td = datetime.strptime(target_date, '%Y-%m-%d').date()
    year, month = td.year, td.month

    # --- 交易日历 ---
    cal_start = td - timedelta(days=10)
    cal_end = td + timedelta(days=hold_days * 2 + 15)
    all_dates = [d.date() for d in pd.date_range(cal_start, cal_end)]
    try:
        idx_df = S.ak.stock_zh_index_daily(symbol='sh000001')
        idx_df['date'] = pd.to_datetime(idx_df['date']).dt.date
        trade_dates = sorted([d for d in all_dates if d in set(idx_df['date'].tolist())])
    except Exception:
        trade_dates = all_dates
    if td not in set(trade_dates):
        S.log(f"!! {td} 不是交易日 (交易日历含 {trade_dates[:3]} ...)")
        # 回退: 用全量日期
        trade_dates = all_dates
    future = [d for d in trade_dates if d > td][:hold_days]
    S.log(f"交易日: {td}  后续 {hold_days} 个交易日: {[d.strftime('%m-%d') for d in future]}")

    # --- 股票池 ---
    pool = S.get_stock_pool()
    if sample_size and sample_size < len(pool):
        if top:
            pool = pool.sort_values('turnover', ascending=False).head(sample_size).reset_index(drop=True)
            S.log(f"股票池: 成交额 Top-{sample_size}")
        else:
            pool = pool.sample(n=sample_size, random_state=42).reset_index(drop=True)
            S.log(f"股票池: 随机抽样 {sample_size}")

    # --- 预拉数据 (key 与月度回测一致, 尽量命中缓存) ---
    first_dt = datetime(year, month, 1).date()
    if month == 12:
        month_last = datetime(year + 1, 1, 1).date() - timedelta(days=1)
    else:
        month_last = datetime(year, month + 1, 1).date() - timedelta(days=1)
    daily_start = (first_dt - timedelta(days=60)).strftime('%Y-%m-%d')
    daily_end = (month_last + timedelta(days=20)).strftime('%Y-%m-%d')
    S.log(f"日K范围: {daily_start} ~ {daily_end} (对齐月度缓存 key)")

    S.log("预拉 15min K (baostock)...")
    cache_15 = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(S.fetch_15min_bs, row['code'], year, month): row['code']
                   for _, row in pool.iterrows()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="15min"):
            code = futures[fut]
            try:
                df = fut.result(timeout=180)
                if df is not None and not df.empty:
                    cache_15[code] = df
            except Exception:
                pass
    S.log(f"15min 就绪: {len(cache_15)} 只")

    S.log("预拉 日K...")
    cache_daily = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(S.fetch_daily, row['code'], daily_start, daily_end): row['code']
                   for _, row in pool.iterrows()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="日K"):
            code = futures[fut]
            try:
                df = fut.result(timeout=60)
                if df is not None and not df.empty:
                    cache_daily[code] = df
            except Exception:
                pass
    S.log(f"日K 就绪: {len(cache_daily)} 只")

    # --- 当日扫描全部命中 ---
    pool_dict = {row['code']: row['name'] for _, row in pool.iterrows()}
    hits, tracks = [], []
    for code in cache_15.keys():
        if code not in cache_daily or code not in pool_dict:
            continue
        res, tr = simulate_one_symbol(code, pool_dict[code], td,
                                      cache_15[code], cache_daily[code],
                                      trade_dates,
                                      min_decline_pct=min_decline_pct,
                                      hold_days=hold_days)
        if res is not None:
            res['pool_rank'] = pool_dict.get(code, '')
            hits.append(res)
            tracks.extend(tr)
    S.log(f"当日命中信号: {len(hits)} 只")

    # --- 去重 + 按信号根绝对成交量排序取前 N ---
    seen, uniq = set(), []
    for r in hits:
        if r['code'] in seen:
            continue
        seen.add(r['code'])
        uniq.append(r)
    uniq.sort(key=lambda r: (r.get('signal_bar_vol', 0), r['code']), reverse=True)
    for i, r in enumerate(uniq, 1):
        r['day_rank'] = i
        r['selected'] = 1 if i <= max_positions else 0

    S.log(f"入选 Top-{max_positions} (按信号根15min成交量): "
          + ", ".join(f"{r['name']}({r['code']})" for r in uniq[:max_positions]))

    df_trades = pd.DataFrame(uniq)
    df_track = pd.DataFrame(tracks)
    return {
        'target_date': target_date, 'trade_dates': trade_dates,
        'future_dates': future, 'hits': df_trades, 'track': df_track,
        'pool_size': len(pool), 'hit_count': len(uniq),
        'params': {'sample': sample_size, 'top': top, 'max_positions': max_positions,
                   'hold_days': hold_days, 'min_decline_pct': min_decline_pct},
    }


# ============ HTML 报告 ============
def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '—'
    return f"{x:+.2f}%"


def cell_html(pct, highlight=False, exited=False):
    """中国股市配色: 涨红跌绿"""
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return '<td class="na">—</td>'
    color = 'up' if pct > 0 else ('down' if pct < 0 else 'flat')
    cls = ['tdc', color]
    if highlight:
        cls.append('sell')
    if exited:
        cls.append('dim')
    return f'<td class="{" ".join(cls)}">{pct:+.2f}%</td>'


def sparkline(values, width=150, height=36):
    """迷你 SVG 折线: values 是 (pct, is_exit) 序列; 返回 svg 字符串"""
    vals = [v for v, _ in values if v is not None]
    if len(vals) < 2:
        return '<span class="na">数据不足</span>'
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    pad = 4
    n = len(values)
    def x(i): return pad + i * (width - 2 * pad) / (n - 1)
    def y(v): return 6 + (hi - v) / rng * (height - 12)
    pts = []
    for i, (v, _) in enumerate(values):
        if v is not None:
            pts.append(f"{x(i):.1f},{y(v):.1f}")
    path = ' '.join(pts)
    # exit point
    exit_pt = None
    for i, (v, e) in enumerate(values):
        if v is not None and e:
            exit_pt = (x(i), y(v))
    html = (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<line x1="{pad}" y1="{y(0):.1f}" x2="{width-pad}" y2="{y(0):.1f}" stroke="#bbb" stroke-dasharray="3 3" stroke-width="1"/>'
            f'<polyline points="{path}" fill="none" stroke="#2f6fed" stroke-width="1.6"/>')
    for i, (v, e) in enumerate(values):
        if v is None:
            continue
        if e:
            html += f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3.2" fill="#f5a623"/>'
        elif i == len(values) - 1:
            html += f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="2.4" fill="#2f6fed"/>'
    html += '</svg>'
    return html


def build_report(data: dict) -> Path:
    hits = data['hits']
    track = data['track']
    fut = data['future_dates']
    target = data['target_date']
    p = data['params']
    sel = hits[hits['selected'] == 1] if not hits.empty else hits
    others = hits[hits['selected'] == 0] if not hits.empty else pd.DataFrame()
    hold = p['hold_days']
    dates_str = ' / '.join(d.strftime('%m-%d') for d in fut)

    # --- 汇总 stats (仅入选) ---
    if not sel.empty:
        n = len(sel)
        wins = int((sel['pnl'] > 0).sum()); losses = int((sel['pnl'] < 0).sum())
        total = float(sel['pnl'].sum())
        wr = wins / n * 100
        tp = int((sel['exit_reason'].str.contains('止盈')).sum())
        fq = int((sel['exit_reason'].str.contains('强平')).sum())
        avg = total / n
    else:
        n = wins = losses = 0; total = 0.0; wr = 0.0; tp = fq = 0; avg = 0.0
    # 全部命中的对照
    if not hits.empty:
        n2 = len(hits)
        t2 = float(hits['pnl'].sum())
        w2 = int((hits['pnl'] > 0).sum())
        wr2 = w2 / n2 * 100
    else:
        n2 = t2 = w2 = 0; wr2 = 0.0

    rows = []
    for _, r in sel.iterrows():  # 只呈现按量排序入选的前 N 只
        tr = track[track['code'] == r['code']].sort_values('day_no')
        tr_map = {int(t['day_no']): t for _, t in tr.iterrows()}
        cells = []
        spark = []
        for dno in range(1, hold + 1):
            t = tr_map.get(dno)
            if t is None:
                cells.append('<td class="na">—</td>')
                spark.append((None, False))
                continue
            cp = t['day_cum_pct']
            is_exit = t['action'] in (f'T+{dno}止盈卖出',) or ('止盈' in str(t['action']) and t['day_no'] == dno)
            exited_before = any(
                (tr_map.get(j) is not None)
                and ('止盈' in str(tr_map[j]['action']) or '强平' in str(tr_map[j]['action']))
                for j in range(1, dno))
            is_exit = ('止盈' in str(t['action']) or '强平' in str(t['action']))
            cells.append(cell_html(cp, highlight=is_exit, exited=exited_before and not is_exit))
            spark.append((cp, is_exit))
        badge = '<span class="badge">买入</span>' if r['selected'] == 1 else ''
        reason = r['exit_reason']
        reason_c = 'sell' if '止盈' in reason else ('flat2' if '强平' in reason else '')
        rows.append(f"""<tr>
<td class="c">{r['day_rank']}</td>
<td><b>{r['name']}</b><div class="sub">{r['code']}</div></td>
<td>{r['date']}</td><td>{str(r['signal_time'])[11:16]}</td>
<td class="num">{r['buy_price']:.2f}</td>
<td class="num">{r['bar_vol_ratio']:.1f}×</td>
<td class="num">{r['decline_cum_pct']:+.1f}%</td>
{''.join(cells)}
<td class="spark">{sparkline(spark)}</td>
<td class="num">{r['buy_price']*1.01:.2f}</td>
<td class="c">{r['exit_date']} </td>
<td class="num">{r['exit_price']:.2f}</td>
<td class="{reason_c}"><b>{reason}</b></td>
<td class="num up-down {'up' if r['pnl_pct']>0 else ('down' if r['pnl_pct']<0 else 'flat')}">{r['pnl_pct']:+.2f}%</td>
<td class="num up-down {'up' if r['pnl']>0 else ('down' if r['pnl']<0 else 'flat')}">{r['pnl']:+,.0f}元</td>
<td>{badge}</td></tr>""")

    # 其余命中仅作一行文字对照 (用户要求: 结果只呈现按量排序的前 N 只)
    others_note = ''
    if len(others):
        others_note = (f"<p class='legend'>当日全池共命中 <b>{len(hits)}</b> 只信号股，"
                       f"这里只呈现其中信号根 15min 成交量最大的 <b>{len(sel)}</b> 只。"
                       f"（其余 {len(others)} 只若按同一规则交易: 合计 {t2:+,.0f} 元、胜率 {wr2:.0f}%，仅作参照，不列入明细。）</p>")

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>单日精确回测 {target} — 15min 放量异动 × 5日跟踪</title>
<style>
:root {{ --up:#d4352c; --down:#0a8f4e; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; margin:0; background:#f5f6f8; color:#1f2430; }}
.wrap {{ max-width:1500px; margin:0 auto; padding:24px; }}
h1 {{ font-size:22px; margin:0 0 4px; }}
h2 {{ font-size:16px; margin:28px 0 10px; padding-left:8px; border-left:4px solid #2f6fed; }}
.sub {{ color:#8a93a6; font-size:11px; font-weight:400; }}
.meta {{ color:#8a93a6; font-size:13px; margin-bottom:16px; }}
.cards {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; }}
.card {{ background:#fff; border:1px solid #e6e8ef; border-radius:10px; padding:14px 18px; min-width:130px; }}
.card .v {{ font-size:24px; font-weight:700; }}
.card .k {{ font-size:12px; color:#8a93a6; margin-top:2px; }}
.card .v.up {{ color:var(--up); }} .card .v.down {{ color:var(--down); }}
table {{ border-collapse:collapse; width:100%; background:#fff; font-size:12.5px; }}
th, td {{ border:1px solid #e6e8ef; padding:5px 6px; text-align:center; white-space:nowrap; }}
th {{ background:#f0f2f7; font-weight:600; position:sticky; top:0; }}
td.num {{ font-variant-numeric:tabular-nums; }}
td.c {{ font-variant-numeric:tabular-nums; }}
.tdc {{ font-variant-numeric:tabular-nums; font-weight:600; }}
.up {{ color:var(--up); }} .down {{ color:var(--down); }} .flat {{ color:#9aa0ac; }}
.up-down {{ font-weight:700; }}
.sell {{ background:#fff7e0; box-shadow: inset 0 0 0 1.5px #f5a623; border-radius:3px; }}
.dim {{ opacity:.45; }}
.na {{ color:#c5c9d3; }}
.badge {{ display:inline-block; background:#d6341f; color:#fff; font-size:10px; padding:2px 7px; border-radius:8px; font-weight:700; }}
.rule {{ background:#fff; border:1px solid #e6e8ef; border-radius:10px; padding:14px 18px; font-size:13px; line-height:1.9; color:#3a4150; }}
.rule b {{ color:#1f2430; }}
.legend {{ font-size:12px; color:#8a93a6; margin:6px 0; }}
.legend i {{ display:inline-block; width:12px;height:12px;border-radius:3px; vertical-align:-1px; margin:0 4px 0 12px; }}
.legend .l-sell {{ background:#fff7e0; box-shadow: inset 0 0 0 1.5px #f5a623; }}
.foot {{ color:#a0a6b5; font-size:12px; margin:24px 0; }}
</style></head><body><div class="wrap">

<h1>单日精确回测：{target} 提取个股 → {hold} 个交易日内逐日对照</h1>
<div class="meta">股票池：成交额 Top-{p['sample']}（{data['pool_size']}只入池快照） ｜ 命中 {data['hit_count']} 只 ｜
规则：近5日累计跌 ≥ {abs(p['min_decline_pct']):.1f}% + 15min 放量≥2× + 涨幅0~2% + 排除三根bar ｜ T+1 起逐日检查，日K收盘≥买入价×1.01 即卖</div>

<div class="cards">
<div class="card"><div class="v {'' if n else ''}">{n}</div><div class="k">入选买入数 (量最大前{p['max_positions']})</div></div>
<div class="card"><div class="v up">{wins}</div><div class="k">盈利 {wins}/{n}</div></div>
<div class="card"><div class="v down">{losses}</div><div class="k">亏损</div></div>
<div class="card"><div class="v">{wr:.0f}%</div><div class="k">入选胜率</div></div>
<div class="card"><div class="v {'up' if total>0 else ('down' if total<0 else '')}">{total:+,.0f}</div><div class="k">入选合计盈亏(元)</div></div>
<div class="card"><div class="v">{avg:+.1f}%</div><div class="k">入选单笔平均</div></div>
<div class="card"><div class="v">{tp}</div><div class="k">触发+1%止盈</div></div>
<div class="card"><div class="v">{fq}</div><div class="k">{hold}日强平</div></div>
</div>

<h2>① 规则速览</h2>
<div class="rule">
<b>信号根(15min)：</b>涨幅 {S.INTRADAY_PCT_MIN:.0f}%~{S.INTRADAY_PCT_MAX:.0f}% ＋ 成交量 &gt; 前一根 ×{S.VOL_MULT:.0f} ＋ 当日累计量/昨日量 ≥ {S.YESTERDAY_VOL_RATIO:.0f}；<b>大前提：</b>近 {S.DECLINE_WINDOW} 个交易日累计收盘跌幅 ≥ {abs(S.MIN_DECLINE_PCT):.1f}%（低位启动）；<b>排除：</b>11:15-11:30 / 13:00-13:15 / 14:45-15:00 三根；<b>入场：</b>信号根收盘价；<b>出场：</b>T+1~T+{hold} 日K收盘价 ≥ 买入价+1% 即当日收盘卖出，否则 T+{hold} 收盘强平。
<div class="legend"><i class="l-sell"></i>金色高亮 = 实际卖出日（止盈/强平），卖出后灰显的格子为“若继续持有”的观察值</div>
</div>

<h2>② 入选 {n} 只（当日信号根 15min 绝对成交量最大）· 逐日收益对照</h2>
<p class="legend">逐日数值 = 当日收盘价相对买入价的累计涨跌（买入后第1个交易日为 T+1）。T+1~T+{hold} 对应日期：{dates_str}</p>
<table>
<tr>
<th>#</th><th>名称</th><th>买入日</th><th>信号bar</th><th>买入价</th><th>量比</th><th>5日跌幅</th>
{' '.join(f'<th>T+{i}<br>{d.strftime("%m-%d")}</th>' for i, d in enumerate(fut, 1))}
<th>轨迹</th><th>止盈线</th><th>卖出日</th><th>卖出价</th><th>出场依据</th><th>收益率</th><th>盈亏</th><th></th>
</tr>
{''.join(rows)}
</table>

<h2>③ 结论</h2>
<div class="rule">
<b>{target} 盘中按规则提取的 {n} 只</b>（信号根 15min 成交量最大者）：
<b>{tp}</b> 只在 {hold} 个交易日内触及 +1% 止盈线并当日收盘卖出，<b>{fq}</b> 只持有至 T+{hold} 强平；
合计盈亏 <b class="{'up' if total>0 else ('down' if total<0 else 'flat')}">{total:+,.0f} 元</b>，
占投入本金（{n}×1 万 = {n*10000:,} 元）的 <b class="{'up' if total>0 else ('down' if total<0 else 'flat')}">{total/(n*10000)*100 if n else 0:+.2f}%</b>，
单笔平均 <b>{avg:+.0f} 元</b>，胜率 <b>{wr:.0f}%</b>。
<div class="legend">{others_note}</div>
</div>

<div class="foot">数据：日K/15min 前复权（qfq）。买入价=信号根收盘，按每只1万元、整百股模拟；A股 T+1，止盈在达到条件的当日收盘确认卖出。仅供研究，非投资建议。</div>
</div></body></html>"""

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"case_{target}_top{data['params']['sample']}_hold{hold}.html"
    f.write_text(html, encoding='utf-8')
    return f


def main():
    ap = argparse.ArgumentParser(description='单日精确案例回测')
    ap.add_argument('--date', default='2026-08-04')
    ap.add_argument('--sample', type=int, default=500)
    ap.add_argument('--no-top', action='store_true', help='不用成交额Top, 改随机抽样')
    ap.add_argument('--max-positions', type=int, default=5)
    ap.add_argument('--hold-days', type=int, default=5)
    ap.add_argument('--min-decline-pct', type=float, default=None)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    md = S.MIN_DECLINE_PCT if args.min_decline_pct is None else args.min_decline_pct
    S.log(f"== 单日精确案例 {args.date} | 池 Top-{args.sample} | 每日买 {args.max_positions} 只 | 持有 {args.hold_days} 日 | 大前提 {md}% ==")
    t0 = time.time()
    data = run_case(args.date, args.sample, top=not args.no_top,
                    max_positions=args.max_positions, hold_days=args.hold_days,
                    max_workers=args.workers, min_decline_pct=md)
    hits = data['hits']
    S.log(f"耗时 {time.time()-t0:.0f}s")

    # CSV 输出
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.date}_top{args.sample}_hold{args.hold_days}"
    if not hits.empty:
        sel = hits[hits['selected'] == 1]
        f_trades = OUT / f"case_{tag}_top{args.max_positions}_trades.csv"
        sel.to_csv(f_trades, index=False, encoding='utf-8-sig')
        S.log(f"前{args.max_positions}只明细: {f_trades}")
        if len(hits) > len(sel):
            f_all = OUT / f"case_{tag}_allhits.csv"
            hits.to_csv(f_all, index=False, encoding='utf-8-sig')
        if not data['track'].empty:
            f_track = OUT / f"case_{tag}_top{args.max_positions}_track.csv"
            data['track'][data['track']['code'].isin(sel['code'])].to_csv(f_track, index=False, encoding='utf-8-sig')
            S.log(f"逐日跟踪: {f_track}")
        s = S.summarize(sel)
        S.print_summary(s)
        s_all = S.summarize(hits)
        print(f"\n[对照] 若当日全部 {len(hits)} 只都买: 胜率 {s_all['win_rate']}% 总盈亏 {s_all['total_pnl']:,.0f} 元")
        # 打印逐日 wide 表
        if not sel.empty:
            fut = data['future_dates']
            wide_rows = []
            for _, r in sel.iterrows():
                tr = data['track'][data['track']['code'] == r['code']].sort_values('day_no')
                row = {'名称': r['name'], '代码': r['code'], '买入价': r['buy_price'],
                       '出场': r['exit_reason'], '收益%': r['pnl_pct'], '盈亏': r['pnl']}
                for _, t in tr.iterrows():
                    row[f"T+{int(t['day_no'])}({str(t['date'])[5:]})"] = (
                        f"{t['day_cum_pct']:+.2f}%" if t['day_cum_pct'] is not None else '-')
                wide_rows.append(row)
            w = pd.DataFrame(wide_rows)
            print("\n" + "=" * 120)
            print(f"  入选股逐日对照 (累计涨跌%)")
            print("=" * 120)
            print(w.to_string(index=False))
    else:
        S.log("当日无任何命中。")
        # 仍然输出空报告
    report = build_report(data)
    S.log(f"HTML 报告: {report}")


if __name__ == '__main__':
    main()

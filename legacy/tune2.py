"""
算法提准 第二轮: 最优组合 + 止损测试
=====================================
基于 tune.py 的发现:
  - 大盘收盘在 MA20 上方: PF 1.81 -> 2.58
  - 近5日跌幅越深越好: <=-3% PF 2.51, <=-5% PF 3.71
  - 量比提高反而变差 (保持 >=2)
  - 下午信号优于上午

本轮:
  1. 测这些条件的组合 (含分月, 检查是否只是某个特殊月贡献)
  2. 用缓存日K重放, 测试加入止损后的效果 (强平 106 笔 -34,851 元是最大拖累)
"""
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import screener_v2 as S

TRADES = Path('results_cap5_v2/backtest_hold5_202408_202508_202608_trades.csv')
TP = 1.01          # 止盈 +1%
HOLD = 5


def stats(d, label):
    if d.empty:
        return {'条件': label, '笔数': 0, '胜率%': 0, '总盈亏': 0, 'PF': 0, '均笔': 0}
    w = d[d['pnl'] > 0]['pnl'].sum()
    l = -d[d['pnl'] < 0]['pnl'].sum()
    return {'条件': label, '笔数': len(d),
            '胜率%': round(len(d[d['pnl'] > 0]) / len(d) * 100, 1),
            '总盈亏': round(d['pnl'].sum(), 0),
            'PF': round(w / l, 2) if l > 0 else 999,
            '均笔': round(d['pnl'].mean(), 0)}


def load_daily_cached(code, month):
    """按月份用与回测一致的 key 读缓存日K"""
    y, m = int(month[:4]), int(month[5:7])
    if m == 12:
        last = pd.Timestamp(y + 1, 1, 1) - pd.Timedelta(days=1)
    else:
        last = pd.Timestamp(y, m + 1, 1) - pd.Timedelta(days=1)
    start = (pd.Timestamp(y, m, 1) - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    end = (last + pd.Timedelta(days=20)).strftime('%Y-%m-%d')
    return S.fetch_daily(code, start, end)


def replay_with_stop(trades: pd.DataFrame, stop_pct: float) -> pd.DataFrame:
    """重放每笔交易: 加入 -stop_pct% 止损 (盘中最低价触及即止损), 其余规则不变"""
    needed = {}
    for _, r in trades.iterrows():
        needed.setdefault(r['code'], set()).add(r['month'])
    cache = {}
    jobs = [(c, m) for c, ms in needed.items() for m in ms]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(load_daily_cached, c, m): (c, m) for c, m in jobs}
        for f in as_completed(futs):
            c, m = futs[f]
            try:
                df = f.result()
                if df is not None and not df.empty:
                    cache[(c, m)] = df
            except Exception:
                pass

    out = []
    for _, r in trades.iterrows():
        ddf = cache.get((r['code'], r['month']))
        buy = float(r['buy_price'])
        shares = int(r['shares'])
        stop_price = buy * (1 - stop_pct / 100.0)
        tp_price = buy * TP
        if ddf is None or ddf.empty:
            out.append({**r.to_dict(), 'pnl_new': r['pnl'], 'reason_new': '无数据'})
            continue
        ddf = ddf.copy()
        ddf['_d'] = pd.to_datetime(ddf['日期']).dt.date
        bd = pd.to_datetime(r['date']).date()
        fut = sorted([d for d in ddf['_d'] if d > bd])[:HOLD]
        rowmap = {row['_d']: row for _, row in ddf.iterrows()}
        pnl_new, reason_new, exit_d = None, None, None
        for off, d in enumerate(fut, 1):
            row = rowmap.get(d)
            if row is None:
                continue
            lo, cl = float(row['最低']), float(row['收盘'])
            if lo <= stop_price:                       # 盘中先触及止损
                pnl_new = (stop_price - buy) * shares
                reason_new, exit_d = f'T+{off}止损(-{stop_pct}%)', d
                break
            if cl >= tp_price:
                pnl_new = (cl - buy) * shares
                reason_new, exit_d = f'T+{off}止盈(+1%)', d
                break
        if pnl_new is None:                            # 强平
            for d in reversed(fut):
                row = rowmap.get(d)
                if row is not None:
                    pnl_new = (float(row['收盘']) - buy) * shares
                    reason_new, exit_d = f'{HOLD}日强平', d
                    break
        if pnl_new is None:
            pnl_new, reason_new = r['pnl'], '缺数据'
        out.append({**r.to_dict(), 'pnl_new': round(pnl_new, 2), 'reason_new': reason_new,
                    'exit_date_new': str(exit_d)})
    return pd.DataFrame(out)


def main():
    d = pd.read_csv(TRADES)
    d['dt'] = pd.to_datetime(d['date']).dt.date
    d['hour'] = pd.to_datetime(d['signal_time']).dt.hour

    idx = S.ak.stock_zh_index_daily(symbol='sh000001')
    idx['date'] = pd.to_datetime(idx['date']).dt.date
    idx = idx.sort_values('date')
    idx['ma20'] = idx['close'].rolling(20).mean()
    idx['above'] = idx['close'] > idx['ma20']
    idx['idx_5d'] = idx['close'].pct_change(5) * 100
    imap = idx.set_index('date')
    d['above'] = d['dt'].map(imap['above'])

    # ---------- 1. 组合 ----------
    combos = [
        ('F1 MA20上方 + 跌幅<=-3%', (d['above'] == True) & (d['decline_cum_pct'] <= -3)),
        ('F2 MA20上方 + 跌幅<=-2%', (d['above'] == True) & (d['decline_cum_pct'] <= -2)),
        ('F3 MA20上方 + 跌幅<=-1%', (d['above'] == True) & (d['decline_cum_pct'] <= -1)),
        ('F4 MA20上方 + 下午', (d['above'] == True) & (d['hour'] >= 13)),
        ('F5 MA20上方 + 跌幅<=-3% + 下午',
         (d['above'] == True) & (d['decline_cum_pct'] <= -3) & (d['hour'] >= 13)),
        ('F6 跌幅<=-3% + 下午', (d['decline_cum_pct'] <= -3) & (d['hour'] >= 13)),
        ('F7 仅跌幅<=-5%', d['decline_cum_pct'] <= -5),
        ('F8 MA20上方 + 跌幅<=-5%', (d['above'] == True) & (d['decline_cum_pct'] <= -5)),
    ]
    print("\n" + "=" * 100)
    print("  1. 增强过滤组合")
    print("=" * 100)
    rows = [stats(d, '基线')] + [stats(d[m], lb) for lb, m in combos]
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n  分月稳定性 (防止只是某个月贡献):")
    print(f"  {'组合':<28}{'2024-08':>22}{'2025-08':>22}{'2026-08':>22}")
    for lb, m in [('基线', pd.Series([True] * len(d)))] + combos:
        line = f"  {lb:<28}"
        for mo in ['2024-08', '2025-08', '2026-08']:
            sub = d[m & (d['month'] == mo)]
            if sub.empty:
                line += f"{'—':>22}"
            else:
                line += f"{len(sub)}笔/{sub['pnl'].sum():>8,.0f}元".rjust(22)
        print(line)

    # ---------- 2. 止损 ----------
    print("\n" + "=" * 100)
    print("  2. 加入止损后重放 (基线: 不加止损, T+1~T+5 检查, 先止损后止盈)")
    print("=" * 100)
    stop_rows = []
    base = replay_with_stop(d, 999)          # 无止损 (校验用)
    for sp in (2, 3, 4, 5):
        rp = replay_with_stop(d, sp)
        rp['pnl'] = rp['pnl_new']
        stop_rows.append(stats(rp, f'止损 -{sp}%'))
        if sp == 3:
            rp.to_csv('results_tune/replay_stop3.csv', index=False, encoding='utf-8-sig')
            print("\n   [止损-3% 出场构成]")
            print(rp.groupby('reason_new')['pnl_new'].agg(['count', 'sum', 'mean']).round(0).to_string())
    print("\n" + pd.DataFrame([stats(d, '基线(无止损)')] + stop_rows).to_string(index=False))

    # ---------- 3. 最优组合 + 止损 ----------
    print("\n" + "=" * 100)
    print("  3. 增强过滤 + 止损-3% 联合")
    print("=" * 100)
    rp = replay_with_stop(d, 3)
    rp['pnl'] = rp['pnl_new']
    rp['above'] = rp['dt'].map(imap['above'])
    joint = []
    for lb, m in [('基线', pd.Series([True] * len(rp)))] + [
            ('F1 MA20上方+跌幅<=-3%', (rp['above'] == True) & (rp['decline_cum_pct'] <= -3)),
            ('F2 MA20上方+跌幅<=-2%', (rp['above'] == True) & (rp['decline_cum_pct'] <= -2)),
            ('F5 MA20上方+跌幅<=-3%+下午',
             (rp['above'] == True) & (rp['decline_cum_pct'] <= -3) & (rp['hour'] >= 13))]:
        joint.append(stats(rp[m], lb))
    print(pd.DataFrame(joint).to_string(index=False))

    Path('results_tune').mkdir(exist_ok=True)
    pd.DataFrame(rows + stop_rows).to_csv('results_tune/tune2_summary.csv', index=False, encoding='utf-8-sig')
    print("\n已保存 results_tune/tune2_summary.csv")


if __name__ == '__main__':
    main()

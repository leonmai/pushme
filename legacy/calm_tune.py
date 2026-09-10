"""
平静度调优扫描 (calm_tune)
==========================
目的: 用户偏好"前面量能平静且均匀, 然后突然爆量"的形态 (京东方A 2026-08-04 13:45 范例)。
     现有 results_cap5_v2 数据缺"前 N 根量", 且已用今/昨>=0.9 过滤过, 无法测该条件必要性。
     本脚本重新扫描, 保留全量原始特征, 供离线对比各条件组合。

产出: results_calm/signals_<yyyymm>.csv  (每根信号一行, 含平静度/爆发倍数/今昨比/后续收益)
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import screener_v2 as S

OUT = Path('results_calm')
OUT.mkdir(exist_ok=True)

CALM_N = (3, 5)          # 平静期窗口 (信号根之前 N 根)


def find_signals_loose(df15: pd.DataFrame, target_day) -> list:
    """宽松找信号: 仅要求 涨幅0~2% + 量>=前根x2, 附加平静度特征 (不做今/昨过滤)"""
    df = df15[~df15['day'].apply(S.is_excluded_bar)].reset_index(drop=True)
    for c in ('open', 'close', 'volume'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['open', 'close', 'volume'])
    days = sorted(df['day'].dt.date.unique())
    if target_day not in days:
        return []
    prev_days = [d for d in days if d < target_day]
    if not prev_days:
        return []
    prev_day = prev_days[-1]
    dv = df.groupby(df['day'].dt.date)['volume'].sum()
    yv = float(dv.get(prev_day, 0))
    tv = float(dv.get(target_day, 0))
    ytd = tv / yv if yv > 0 else 0.0

    bars = df[df['day'].dt.date == target_day].reset_index(drop=True)
    out = []
    for i in range(1, len(bars)):
        cur, prev = bars.iloc[i], bars.iloc[i - 1]
        op, cl = float(cur['open']), float(cur['close'])
        cv, pv = float(cur['volume']), float(prev['volume'])
        if op <= 0 or pv <= 0:
            continue
        chg = (cl - op) / op * 100
        vr = cv / pv
        if not (S.INTRADAY_PCT_MIN <= chg <= S.INTRADAY_PCT_MAX):
            continue
        if vr < S.VOL_MULT:
            continue
        f = {'close': cl, 'bar_change_pct': chg,
             'vol_ratio': vr, 'signal_bar_vol': cv, 'prev_bar_vol': pv,
             'ytd_vol_ratio': ytd, 'bar_idx': i}
        for n in CALM_N:
            if i >= n:
                vs = [float(bars.iloc[j]['volume']) for j in range(i - n, i)]
                mn, mx, av = min(vs), max(vs), sum(vs) / n
                f[f'calm{n}_maxmin'] = mx / mn if mn > 0 else 99.0
                f[f'calm{n}_avg'] = av
                f[f'burst{n}'] = cv / av if av > 0 else 0.0
            else:
                f[f'calm{n}_maxmin'] = np.nan
                f[f'calm{n}_avg'] = np.nan
                f[f'burst{n}'] = np.nan
        f['signal_time'] = str(cur['day'])
        out.append(f)
    return out


def simulate_exit(ddf: pd.DataFrame, buy_dt, buy_price: float,
                  hold: int = 5, tp: float = 1.0) -> dict:
    """T+1 起 hold 个交易日内, 日K收盘 >= 买入价*(1+tp%) 即按当日收盘卖出; 否则末日强平"""
    ddf = ddf.copy()
    ddf['日期'] = pd.to_datetime(ddf['日期']).dt.date
    fut = ddf[ddf['日期'] > buy_dt].head(hold)
    if fut.empty:
        return {}
    for i, (_, r) in enumerate(fut.iterrows(), 1):
        c = float(r['收盘'])
        if c >= buy_price * (1 + tp / 100):
            return {'exit_date': str(r['日期']), 'exit_price': c, 'hold_days': i,
                    'pnl_pct': (c / buy_price - 1) * 100, 'exit_reason': '止盈'}
    last = fut.iloc[-1]
    c = float(last['收盘'])
    return {'exit_date': str(last['日期']), 'exit_price': c, 'hold_days': len(fut),
            'pnl_pct': (c / buy_price - 1) * 100, 'exit_reason': '强平'}


def scan_month(year: int, month: int, sample: int, workers: int):
    S.log(f'=== 扫描 {year}-{month:02d} ===')
    pool = S.get_stock_pool()
    pool = pool[pool['code'].apply(S.is_valid_pool_code)]
    pool = pool.sort_values('turnover', ascending=False).head(sample)
    codes = [(r['code'], r['name']) for _, r in pool.iterrows()]
    S.log(f'池子 {len(codes)} 只')

    cal = S.fetch_daily('000001', f'{year}-{month:02d}-01', f'{year}-{month:02d}-28')
    if cal.empty:
        S.log('日历获取失败')
        return pd.DataFrame()
    cal['日期'] = pd.to_datetime(cal['日期'])
    days = [d.date() for d in cal['日期'] if d.month == month]
    S.log(f'交易日 {len(days)} 天')

    start = f'{year}-{month:02d}-01'
    end = (pd.Timestamp(f'{year}-{month:02d}-01') + pd.offsets.MonthEnd(0)
           + timedelta(days=25)).strftime('%Y-%m-%d')

    def work(item):
        code, name = item
        ddf = S.fetch_daily(code, start, end)
        if ddf.empty:
            return []
        ddf = ddf.copy()
        ddf['日期'] = pd.to_datetime(ddf['日期'])
        df15 = S.fetch_15min_bs(code, year, month)
        if df15 is None or df15.empty:
            return []
        df15 = df15.copy()
        df15['day'] = pd.to_datetime(df15['day'])
        rows = []
        for d in days:
            ok, diag = S.check_recent_decline(ddf, d, min_pct=99)
            if not diag:
                continue
            cum = diag['cum_pct']
            if cum > -1.0:              # 大前提: 近5日累计下跌 > 1%
                continue
            for f in find_signals_loose(df15, d):
                ex = simulate_exit(ddf, d, f['close'])
                if not ex:
                    continue
                rows.append({'code': code, 'name': name, 'date': str(d),
                             'decline_cum_pct': cum, **f, **ex})
        return rows

    all_rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, c) for c in codes]
        done = 0
        for fu in as_completed(futs):
            done += 1
            if done % 100 == 0:
                S.log(f'  进度 {done}/{len(codes)}')
            try:
                all_rows.extend(fu.result())
            except Exception:
                pass
    df = pd.DataFrame(all_rows)
    S.log(f'共 {len(df)} 条信号')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', default='2026-08')
    ap.add_argument('--sample', type=int, default=500)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    for m in args.months.split(','):
        y, mm = int(m[:4]), int(m[5:7])
        df = scan_month(y, mm, args.sample, args.workers)
        if not df.empty:
            f = OUT / f'signals_{y}{mm:02d}.csv'
            df.to_csv(f, index=False, encoding='utf-8-sig')
            S.log(f'保存 {f}')


if __name__ == '__main__':
    main()

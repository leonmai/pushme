"""收盘验证: 今日信号股从信号价到收盘的实际走势"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import live_scout as L

TODAY = '2026-09-07'


def main():
    d = pd.read_csv('results_live/signals_2026-09-07.csv')
    d['code'] = d['code'].astype(str).str.zfill(6)
    d = d.drop_duplicates(subset=['code', 'bar_time']).reset_index(drop=True)
    print(f'去重后 {len(d)} 条信号\n')

    rows = []
    for _, r in d.iterrows():
        df = L.fetch_15min_live(r['code'])
        if df.empty:
            continue
        df['day'] = pd.to_datetime(df['day'])
        today = df[df['day'].dt.strftime('%Y-%m-%d') == TODAY]
        if today.empty:
            continue
        for c in ('open', 'high', 'low', 'close', 'volume'):
            today[c] = pd.to_numeric(today[c], errors='coerce')
        sig_t = pd.to_datetime(r['bar_time'])
        after = today[today['day'] > sig_t]
        if after.empty:
            rows.append({
                'code': r['code'], 'name': r['name'], 'bar': str(sig_t)[11:16],
                'sig': r['close'], 'close': float(today['close'].iloc[-1]),
                'ret': 0.0, 'max': 0.0, 'min': 0.0, 'bars': 0,
                'vol_ratio': r['vol_ratio'], 'calm': r['calm3_maxmin'],
                'burst': r['burst3'], 'grade': r['grade'],
            })
            continue
        sig = float(r['close'])
        hi = float(after['high'].max())
        lo = float(after['low'].min())
        cl = float(today['close'].iloc[-1])
        rows.append({
            'code': r['code'], 'name': r['name'], 'bar': str(sig_t)[11:16],
            'sig': sig, 'close': cl,
            'ret': (cl - sig) / sig * 100,
            'max': (hi - sig) / sig * 100,
            'min': (lo - sig) / sig * 100,
            'bars': len(after),
            'vol_ratio': r['vol_ratio'], 'calm': r['calm3_maxmin'],
            'burst': r['burst3'], 'grade': r['grade'],
        })

    res = pd.DataFrame(rows)
    if res.empty:
        print('无数据')
        return
    res = res.sort_values('ret', ascending=False).reset_index(drop=True)
    res.to_csv('results_live/verify_close_2026-09-07.csv', index=False,
               encoding='utf-8-sig')

    n = len(res)
    win = (res['ret'] > 0).sum()
    avg = res['ret'].mean()
    # 触及 +1% 的比例(模拟次日前的日内机会)
    tp = (res['max'] >= 1.0).sum()

    print('=' * 96)
    print(f'  今日 {n} 只信号股: 信号价 → 收盘')
    print('=' * 96)
    show = res.copy()
    show['sig'] = show['sig'].round(2)
    show['close'] = show['close'].round(2)
    show['ret'] = show['ret'].round(2)
    show['max'] = show['max'].round(2)
    show['min'] = show['min'].round(2)
    show['vol_ratio'] = show['vol_ratio'].round(2)
    show['calm'] = show['calm'].round(2)
    show['burst'] = show['burst'].round(2)
    show.columns = ['代码', '名称', '信号bar', '信号价', '收盘', '收盘涨跌%',
                    '盘中最高%', '盘中最低%', '后续bar数', '量比', '平静度',
                    '爆发', '级别']
    print(show.to_string(index=False))
    print()
    print(f'  收盘为正: {win}/{n} = {win / n * 100:.1f}%')
    print(f'  平均收盘涨跌: {avg:+.2f}%')
    print(f'  盘中曾触及 +1%: {tp}/{n} = {tp / n * 100:.1f}%')

    # 平静蓄势组 vs 其余
    star = res[(res['calm'] <= 1.5) & (res['burst'] >= 2.0)]
    other = res[~((res['calm'] <= 1.5) & (res['burst'] >= 2.0))]
    if len(star) and len(other):
        print()
        print(f'  ★ 平静蓄势组 (平静≤1.5 且 爆发≥2): {len(star)} 只, '
              f'平均 {star["ret"].mean():+.2f}%, 正收益 {(star["ret"] > 0).sum()}/{len(star)}')
        print(f'    其余: {len(other)} 只, 平均 {other["ret"].mean():+.2f}%, '
              f'正收益 {(other["ret"] > 0).sum()}/{len(other)}')

    # 量比前5
    print()
    print('  量比前5:')
    for _, r in res.nlargest(5, 'vol_ratio').iterrows():
        print(f'    {r["name"]:<8} {r["code"]}  量比{r["vol_ratio"]:.2f}×  '
              f'信号价{r["sig"]:.2f} → 收盘{r["close"]:.2f}  {r["ret"]:+.2f}%')


if __name__ == '__main__':
    main()

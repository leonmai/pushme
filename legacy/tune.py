"""
算法提准: 对三个月 302 笔真实成交做离线参数敏感性分析
======================================================
目的: 在已有成交明细上测试各种"过滤条件", 看哪些条件能真的提升胜率/PF,
      再把最优组合写进盘中监控脚本 live_scout.py。

测试维度:
  A. 大盘环境过滤 (上证指数): 近5日累计涨跌 / 收盘 vs MA20 / 当日涨跌
  B. 信号根量比阈值: >=2(基线) / >=2.5 / >=3
  C. 大前提深度: 近5日累计跌幅 <= -0.5%(基线) / <=-1% / <=-2% / <=-3%
  D. 信号出现时段: 上午 vs 下午
  E. 组合条件
"""
import sys
from pathlib import Path

import akshare as ak
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import screener_v2 as S

TRADES = Path('results_cap5_v2/backtest_hold5_202408_202508_202608_trades.csv')


def load_index():
    df = ak.stock_zh_index_daily(symbol='sh000001')
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.sort_values('date').reset_index(drop=True)
    df['idx_pct'] = df['close'].pct_change() * 100
    df['idx_5d'] = df['close'].pct_change(5) * 100
    df['ma20'] = df['close'].rolling(20).mean()
    df['above_ma20'] = df['close'] > df['ma20']
    return df.set_index('date')


def stats(d: pd.DataFrame, label: str) -> dict:
    if d.empty:
        return {'条件': label, '笔数': 0, '胜率%': 0, '总盈亏': 0, 'PF': 0, '均笔': 0}
    w = d[d['pnl'] > 0]['pnl'].sum()
    l = -d[d['pnl'] < 0]['pnl'].sum()
    return {
        '条件': label,
        '笔数': len(d),
        '胜率%': round(len(d[d['pnl'] > 0]) / len(d) * 100, 1),
        '总盈亏': round(d['pnl'].sum(), 0),
        'PF': round(w / l, 2) if l > 0 else 999,
        '均笔': round(d['pnl'].mean(), 0),
    }


def main():
    d = pd.read_csv(TRADES)
    d['dt'] = pd.to_datetime(d['date']).dt.date
    d['hour'] = pd.to_datetime(d['signal_time']).dt.hour
    idx = load_index()
    d['idx_5d'] = d['dt'].map(idx['idx_5d'])
    d['idx_pct'] = d['dt'].map(idx['idx_pct'])
    d['above_ma20'] = d['dt'].map(idx['above_ma20'])

    rows = [stats(d, '基线 (无过滤)')]
    print(f"\n基线: {len(d)} 笔 胜率 {stats(d,'x')['胜率%']}% 总盈亏 {d['pnl'].sum():,.0f} PF {stats(d,'x')['PF']}\n")

    # --- A 大盘环境 ---
    print("=" * 96)
    print("  A. 大盘环境过滤 (买入日上证指数状态)")
    print("=" * 96)
    a_rows = [
        stats(d[d['idx_5d'] > 0], 'A1 大盘近5日上涨 (idx_5d>0)'),
        stats(d[d['idx_5d'] <= 0], 'A2 大盘近5日下跌 (idx_5d<=0)'),
        stats(d[d['above_ma20'] == True], 'A3 大盘收盘在MA20上方'),
        stats(d[d['above_ma20'] != True], 'A4 大盘收盘在MA20下方'),
        stats(d[d['idx_pct'] > 0], 'A5 大盘当日上涨'),
    ]
    rows += a_rows
    print(pd.DataFrame(a_rows).to_string(index=False))
    print("\n  分月看大盘过滤效果:")
    for m in sorted(d['month'].unique()):
        dm = d[d['month'] == m]
        s0 = stats(dm, m)
        s1 = stats(dm[dm['idx_5d'] > 0], m + ' + 大盘5日上涨')
        print(f"    {m}: 基线 {s0['笔数']}笔 {s0['总盈亏']:>9,.0f}元 PF{s0['PF']:<5} "
              f"→ 过滤后 {s1['笔数']}笔 {s1['总盈亏']:>9,.0f}元 PF{s1['PF']}")

    # --- B 量比阈值 ---
    print("\n" + "=" * 96)
    print("  B. 信号根量比阈值")
    print("=" * 96)
    b_rows = [stats(d[d['bar_vol_ratio'] >= t], f'B 量比 >= {t}') for t in (2, 2.5, 3, 4)]
    rows += b_rows
    print(pd.DataFrame(b_rows).to_string(index=False))

    # --- C 大前提深度 ---
    print("\n" + "=" * 96)
    print("  C. 大前提: 近5日累计跌幅深度")
    print("=" * 96)
    c_rows = [stats(d[d['decline_cum_pct'] <= t], f'C 近5日跌幅 <= {t}%') for t in (-0.5, -1, -2, -3, -5)]
    rows += c_rows
    print(pd.DataFrame(c_rows).to_string(index=False))

    # --- D 时段 ---
    print("\n" + "=" * 96)
    print("  D. 信号出现时段")
    print("=" * 96)
    d_rows = [
        stats(d[d['hour'] < 11], 'D1 上午 (9:45-11:15)'),
        stats(d[d['hour'] >= 13], 'D2 下午 (13:30-14:45)'),
        stats(d[(d['hour'] == 9) | ((d['hour'] == 10) & (pd.to_datetime(d['signal_time']).dt.minute <= 15))],
              'D3 早盘 (9:45-10:15)'),
    ]
    rows += d_rows
    print(pd.DataFrame(d_rows).to_string(index=False))

    # --- E 组合 ---
    print("\n" + "=" * 96)
    print("  E. 组合条件")
    print("=" * 96)
    combos = [
        ('E1 大盘5日涨 + 量比>=2.5', (d['idx_5d'] > 0) & (d['bar_vol_ratio'] >= 2.5)),
        ('E2 大盘5日涨 + 跌幅<=-2%', (d['idx_5d'] > 0) & (d['decline_cum_pct'] <= -2)),
        ('E3 大盘MA20上方 + 量比>=2.5', (d['above_ma20'] == True) & (d['bar_vol_ratio'] >= 2.5)),
        ('E4 大盘5日涨 + 跌幅<=-2% + 量比>=2.5',
         (d['idx_5d'] > 0) & (d['decline_cum_pct'] <= -2) & (d['bar_vol_ratio'] >= 2.5)),
        ('E5 大盘5日涨 + 下午信号', (d['idx_5d'] > 0) & (d['hour'] >= 13)),
        ('E6 大盘5日涨 + 跌幅<=-2% + 下午',
         (d['idx_5d'] > 0) & (d['decline_cum_pct'] <= -2) & (d['hour'] >= 13)),
    ]
    e_rows = [stats(d[m], label) for label, m in combos]
    rows += e_rows
    print(pd.DataFrame(e_rows).to_string(index=False))

    out = pd.DataFrame(rows)
    Path('results_tune').mkdir(exist_ok=True)
    out.to_csv('results_tune/filter_sensitivity.csv', index=False, encoding='utf-8-sig')
    print(f"\n已保存: results_tune/filter_sensitivity.csv ({len(out)} 组)")

    best = out[(out['笔数'] >= 60)].sort_values('总盈亏', ascending=False)
    print("\n" + "=" * 96)
    print("  笔数>=60 的组合里, 按总盈亏排序 TOP-6")
    print("=" * 96)
    print(best.head(6).to_string(index=False))


if __name__ == '__main__':
    main()

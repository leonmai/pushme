"""三月合并验证: 排序方式 / 今昨量能 / 平静度 / 早盘信号"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path('.').resolve()))
import screener_v2 as S

CAP = 10000
MONTHS = [('202408', '2024-08'), ('202508', '2025-08'), ('202608', '2026-08')]


def stats(df):
    if df.empty:
        return {'n': 0, 'win': 0, 'pnl': 0, 'pf': 0, 'avg': 0}
    pnl = df['pnl_pct'] / 100 * CAP
    win = (pnl > 0).mean() * 100
    gp, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {'n': len(df), 'win': win, 'pnl': pnl.sum(),
            'pf': gp / gl if gl > 0 else 99.9, 'avg': pnl.mean()}


def top5(df, col, asc=False):
    d = df.sort_values('date').copy()
    d['_rk'] = d.groupby('date')[col].rank(ascending=asc, method='first')
    return d[d['_rk'] <= 5]


def main():
    frames = []
    for tag, lab in MONTHS:
        p = Path(f'results_calm/signals_{tag}.csv')
        if not p.exists():
            continue
        d = pd.read_csv(p).dropna(subset=['pnl_pct'])
        d['month'] = lab
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df['calm3_p'] = df['calm3_maxmin'].fillna(1.0)
    S.log(f'三月合计 {len(df)} 条信号, {df["date"].nunique()} 个交易日\n')

    S.log('=' * 108)
    S.log('A) 排序方式: 绝对成交量 vs 量比倍数  (每日取前5, 每只1万)')
    S.log('=' * 108)
    S.log(f'{"配置":<26}{"月份":<10}{"笔数":>6}{"胜率%":>9}{"总盈亏":>12}{"PF":>8}{"单笔均":>9}')
    S.log('-' * 108)
    for label, col in [('绝对成交量排 (现v6)', 'signal_bar_vol'), ('量比倍数排 (改后)', 'vol_ratio')]:
        for _, lab in MONTHS:
            s = df[df['month'] == lab]
            st = stats(top5(s, col))
            S.log(f'{label:<24}{lab:<10}{st["n"]:>6}{st["win"]:>9.1f}{st["pnl"]:>12,.0f}{st["pf"]:>8.2f}{st["avg"]:>9,.0f}')
        st = stats(top5(df, col))
        S.log(f'{label:<24}{"三月合计":<10}{st["n"]:>6}{st["win"]:>9.1f}{st["pnl"]:>12,.0f}{st["pf"]:>8.2f}{st["avg"]:>9,.0f}')
        S.log('')

    S.log('=' * 108)
    S.log('B) 今/昨量能 (ytd_vol_ratio) 阈值 — 基于"量比排"')
    S.log('=' * 108)
    S.log(f'{"配置":<26}{"月份":<10}{"笔数":>6}{"胜率%":>9}{"总盈亏":>12}{"PF":>8}{"单笔均":>9}')
    S.log('-' * 108)
    for th in (0.0, 0.8, 0.9, 1.0, 1.2):
        s0 = df if th == 0 else df[df['ytd_vol_ratio'] >= th]
        lab0 = '无过滤' if th == 0 else f'今/昨>={th}'
        st = stats(top5(s0, 'vol_ratio'))
        S.log(f'{lab0:<24}{"三月合计":<10}{st["n"]:>6}{st["win"]:>9.1f}{st["pnl"]:>12,.0f}{st["pf"]:>8.2f}{st["avg"]:>9,.0f}')
    S.log('')

    S.log('=' * 108)
    S.log('C) 平静度 calm3_maxmin<=1.5 (京东方A 8/4 实测 1.31) — 基于"量比排 + 今昨>=0.9"')
    S.log('=' * 108)
    S.log(f'{"配置":<34}{"月份":<10}{"笔数":>6}{"胜率%":>9}{"总盈亏":>12}{"PF":>8}{"单笔均":>9}')
    S.log('-' * 108)
    base = df[df['ytd_vol_ratio'] >= 0.9]
    calmp = base[base['calm3_p'] <= 1.5]
    calmq = base.dropna(subset=['calm3_maxmin'])
    calmq = calmq[calmq['calm3_maxmin'] <= 1.5]
    for label, s in [('无平静度过滤', base),
                     ('+ calm<=1.5 (早盘放行)', calmp),
                     ('+ calm<=1.5 (早盘拒绝)', calmq)]:
        for _, lab in MONTHS:
            st = stats(top5(s[s['month'] == lab], 'vol_ratio'))
            S.log(f'{label:<32}{lab:<10}{st["n"]:>6}{st["win"]:>9.1f}{st["pnl"]:>12,.0f}{st["pf"]:>8.2f}{st["avg"]:>9,.0f}')
        st = stats(top5(s, 'vol_ratio'))
        S.log(f'{label:<32}{"三月合计":<10}{st["n"]:>6}{st["win"]:>9.1f}{st["pnl"]:>12,.0f}{st["pf"]:>8.2f}{st["avg"]:>9,.0f}')
        S.log('')

    S.log('=' * 108)
    S.log('D) 早盘信号 (bar_idx<3, 10:15前触发) vs 非早盘 — 三月全量')
    S.log('=' * 108)
    S.log(f'{"分组":<26}{"月份":<10}{"笔数":>6}{"胜率%":>9}{"总盈亏":>12}{"PF":>8}{"单笔均":>9}')
    S.log('-' * 108)
    for label, cond in [('早盘 bar_idx<3', df['bar_idx'] < 3), ('非早盘', df['bar_idx'] >= 3)]:
        s = df[cond]
        for _, lab in MONTHS:
            st = stats(s[s['month'] == lab])
            S.log(f'{label:<24}{lab:<10}{st["n"]:>6}{st["win"]:>9.1f}{st["pnl"]:>12,.0f}{st["pf"]:>8.2f}{st["avg"]:>9,.0f}')
        st = stats(s)
        S.log(f'{label:<24}{"三月合计":<10}{st["n"]:>6}{st["win"]:>9.1f}{st["pnl"]:>12,.0f}{st["pf"]:>8.2f}{st["avg"]:>9,.0f}')
        S.log('')


if __name__ == '__main__':
    main()

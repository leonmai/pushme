"""补测: 统一基准下对比 (量比排 + 今/昨 + 平静度), 并检验早盘信号(calm3缺失)该留还是该弃"""
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path('.').resolve()))
import screener_v2 as S

CAP = 10000


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


def show(tag, st):
    S.log(f'{tag:<44}{st["n"]:>5}{st["win"]:>9.1f}{st["pnl"]:>12,.0f}{st["pf"]:>8.2f}{st["avg"]:>9,.0f}')


def main():
    df = pd.read_csv('results_calm/signals_202608.csv').dropna(subset=['pnl_pct'])
    S.log(f'全量 {len(df)} 条, {df["date"].nunique()} 个交易日\n')

    # 早盘信号 (bar_idx<3, calm3 无值) 单独看表现
    early = df[df['bar_idx'] < 3]
    late = df[df['bar_idx'] >= 3]
    S.log('=' * 100)
    S.log('0) 早盘信号 (bar_idx<3, 即 10:15 前触发, 无法算前3根平静度) 该留还是该弃?')
    S.log('=' * 100)
    S.log(f'{"分组":<44}{"笔数":>5}{"胜率%":>9}{"总盈亏":>12}{"PF":>8}{"单笔均":>9}')
    S.log('-' * 100)
    show('早盘信号 (bar_idx<3) 全部', stats(early))
    show('非早盘信号 (bar_idx>=3) 全部', stats(late))
    show('早盘信号 按量比排取前5/日', stats(top5(early, 'vol_ratio')))
    show('非早盘信号 按量比排取前5/日', stats(top5(late, 'vol_ratio')))
    S.log('')

    # ---- 统一基准对比 ----
    S.log('=' * 100)
    S.log('1) 统一基准: 按量比排取前5/日, 逐层叠加条件  (NaN=早盘, 两种处理)')
    S.log('=' * 100)
    S.log(f'{"配置":<44}{"笔数":>5}{"胜率%":>9}{"总盈亏":>12}{"PF":>8}{"单笔均":>9}')
    S.log('-' * 100)

    show('A0 裸: 量比排, 无任何过滤', stats(top5(df, 'vol_ratio')))
    sub = df[df['ytd_vol_ratio'] >= 0.9]
    show('A1 = A0 + 今/昨>=0.9', stats(top5(sub, 'vol_ratio')))

    # 平静度: 早盘 NaN 视为"通过" (填 1.0 / 大值)
    p = df.copy()
    p['calm3_maxmin_p'] = p['calm3_maxmin'].fillna(1.0)
    p['burst3_p'] = p['burst3'].fillna(99.0)
    for mm in (1.5, 2.0):
        s = p[p['calm3_maxmin_p'] <= mm]
        show(f'A2 = A0 + calm3<={mm} (早盘放行)', stats(top5(s, 'vol_ratio')))
        s = p[(p['calm3_maxmin_p'] <= mm) & (p['ytd_vol_ratio'] >= 0.9)]
        show(f'A3 = A1 + calm3<={mm} (早盘放行)', stats(top5(s, 'vol_ratio')))

    # 平静度: 早盘 NaN 视为"拒绝" (必须有前3根)
    q = df.dropna(subset=['calm3_maxmin', 'burst3'])
    for mm in (1.5, 2.0):
        s = q[q['calm3_maxmin'] <= mm]
        show(f'A4 = A0 + calm3<={mm} (早盘拒绝)', stats(top5(s, 'vol_ratio')))
        s = q[(q['calm3_maxmin'] <= mm) & (q['ytd_vol_ratio'] >= 0.9)]
        show(f'A5 = A1 + calm3<={mm} (早盘拒绝)', stats(top5(s, 'vol_ratio')))
    S.log('')

    # ---- 加 burst ----
    S.log('=' * 100)
    S.log('2) 再叠加 burst3 (信号根量 / 前3根均量) — 京东方A 8/4 参考: calm3=1.31, burst3=2.47')
    S.log('=' * 100)
    S.log(f'{"配置":<44}{"笔数":>5}{"胜率%":>9}{"总盈亏":>12}{"PF":>8}{"单笔均":>9}')
    S.log('-' * 100)
    for mm, b in itertools.product((1.5, 2.0), (2.0, 2.5)):
        s = p[(p['calm3_maxmin_p'] <= mm) & (p['burst3_p'] >= b) & (p['ytd_vol_ratio'] >= 0.9)]
        show(f'今昨>=0.9 + calm<={mm} + burst>={b} (早盘放行)', stats(top5(s, 'vol_ratio')))
    for mm, b in itertools.product((1.5, 2.0), (2.0, 2.5)):
        s = q[(q['calm3_maxmin'] <= mm) & (q['burst3'] >= b) & (q['ytd_vol_ratio'] >= 0.9)]
        show(f'今昨>=0.9 + calm<={mm} + burst>={b} (早盘拒绝)', stats(top5(s, 'vol_ratio')))
    S.log('')

    # ---- 分级: 跌幅深度 x 平静度 ----
    S.log('=' * 100)
    S.log('3) 跌幅深度 与 平静度 交叉 (按量比排取前5/日, 今昨>=0.9)')
    S.log('=' * 100)
    S.log(f'{"配置":<44}{"笔数":>5}{"胜率%":>9}{"总盈亏":>12}{"PF":>8}{"单笔均":>9}')
    S.log('-' * 100)
    for lab, cond in [('跌>1% (B级及以上)', p['decline_cum_pct'] <= -1.0),
                      ('跌>3% (仅A级)', p['decline_cum_pct'] <= -3.0),
                      ('跌>5%', p['decline_cum_pct'] <= -5.0)]:
        s = p[cond & (p['ytd_vol_ratio'] >= 0.9)]
        show(f'{lab} 无平静度', stats(top5(s, 'vol_ratio')))
        s = p[cond & (p['ytd_vol_ratio'] >= 0.9) & (p['calm3_maxmin_p'] <= 1.5)]
        show(f'{lab} + calm3<=1.5', stats(top5(s, 'vol_ratio')))


if __name__ == '__main__':
    main()

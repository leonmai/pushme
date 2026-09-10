"""离线对比: 今/昨量能必要性 + 平静度条件 + 排序方式 (每日取前5, 每只1万)"""
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path('.').resolve()))
import screener_v2 as S

CAP = 10000


def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {'n': 0, 'win': 0, 'pnl': 0, 'pf': 0, 'avg': 0}
    pnl = df['pnl_pct'] / 100 * CAP
    win = (pnl > 0).mean() * 100
    gp, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {'n': len(df), 'win': win, 'pnl': pnl.sum(),
            'pf': gp / gl if gl > 0 else 99.9, 'avg': pnl.mean()}


def top5_by_day(df: pd.DataFrame, sort_col: str, asc: bool = False) -> pd.DataFrame:
    """每个交易日按 sort_col 排序取前 5"""
    d = df.sort_values('date').copy()
    d['_rk'] = d.groupby('date')[sort_col].rank(ascending=asc, method='first')
    return d[d['_rk'] <= 5]


def main():
    df = pd.read_csv('results_calm/signals_202608.csv')
    df = df.dropna(subset=['pnl_pct'])
    S.log(f'全量信号 {len(df)} 条, 交易日 {df["date"].nunique()} 天')
    S.log(f'字段: {df.columns.tolist()}\n')

    # ---------- A) 今/昨量能 是否必要 ----------
    S.log('=' * 96)
    S.log('A) 今/昨量能 (ytd_vol_ratio) 条件的必要性')
    S.log('=' * 96)
    S.log(f'{"配置":<34}{"笔数":>6}{"胜率%":>9}{"总盈亏":>12}{"PF":>8}{"单笔均":>10}')
    S.log('-' * 96)
    base = stats(top5_by_day(df, 'signal_bar_vol'))
    S.log(f'{"全量(无今昨过滤) 按绝对量排":<30}{base["n"]:>6}{base["win"]:>9.1f}{base["pnl"]:>12,.0f}{base["pf"]:>8.2f}{base["avg"]:>10,.0f}')
    for th in (0.6, 0.9, 1.0, 1.2):
        sub = df[df['ytd_vol_ratio'] >= th]
        st = stats(top5_by_day(sub, 'signal_bar_vol'))
        S.log(f'{"今/昨 >= " + str(th) + " 按绝对量排":<30}{st["n"]:>6}{st["win"]:>9.1f}{st["pnl"]:>12,.0f}{st["pf"]:>8.2f}{st["avg"]:>10,.0f}')
    S.log('')

    # ---------- B) 排序方式 ----------
    S.log('=' * 96)
    S.log('B) 排序方式: 信号根绝对成交量  vs  量比倍数 (vol_ratio)')
    S.log('=' * 96)
    S.log(f'{"配置":<34}{"笔数":>6}{"胜率%":>9}{"总盈亏":>12}{"PF":>8}{"单笔均":>10}')
    S.log('-' * 96)
    for label, col in [('按绝对成交量排', 'signal_bar_vol'), ('按量比倍数排', 'vol_ratio')]:
        for tag, sub in [('无今昨过滤', df), ('今/昨>=0.9', df[df['ytd_vol_ratio'] >= 0.9])]:
            st = stats(top5_by_day(sub, col))
            S.log(f'{label + " + " + tag:<30}{st["n"]:>6}{st["win"]:>9.1f}{st["pnl"]:>12,.0f}{st["pf"]:>8.2f}{st["avg"]:>10,.0f}')
    S.log('')

    # ---------- C) 平静度 ----------
    S.log('=' * 96)
    S.log('C) 平静度条件 (calm3 = 信号根前3根): maxmin越小越均匀, burst3=信号根/前3根均量')
    S.log('=' * 96)
    S.log('京东方A 8/4 参考值: calm5_maxmin=1.31, burst5=2.49')
    S.log('')
    S.log(f'{"平静度条件":<40}{"笔数":>6}{"胜率%":>9}{"总盈亏":>12}{"PF":>8}{"单笔均":>10}')
    S.log('-' * 96)
    d3 = df.dropna(subset=['calm3_maxmin', 'burst3'])
    st = stats(top5_by_day(d3, 'signal_bar_vol'))
    S.log(f'{"(基准) 无平静度过滤":<36}{st["n"]:>6}{st["win"]:>9.1f}{st["pnl"]:>12,.0f}{st["pf"]:>8.2f}{st["avg"]:>10,.0f}')
    for mm in (1.5, 2.0, 2.5, 3.0):
        sub = d3[d3['calm3_maxmin'] <= mm]
        st = stats(top5_by_day(sub, 'signal_bar_vol'))
        S.log(f'{"calm3_maxmin <= " + str(mm):<36}{st["n"]:>6}{st["win"]:>9.1f}{st["pnl"]:>12,.0f}{st["pf"]:>8.2f}{st["avg"]:>10,.0f}')
    S.log('')
    for b in (1.5, 2.0, 2.5, 3.0):
        sub = d3[d3['burst3'] >= b]
        st = stats(top5_by_day(sub, 'signal_bar_vol'))
        S.log(f'{"burst3 >= " + str(b):<36}{st["n"]:>6}{st["win"]:>9.1f}{st["pnl"]:>12,.0f}{st["pf"]:>8.2f}{st["avg"]:>10,.0f}')
    S.log('')

    # 组合
    S.log('组合: calm3_maxmin <= M 且 burst3 >= B (按量比排)')
    S.log(f'{"组合":<40}{"笔数":>6}{"胜率%":>9}{"总盈亏":>12}{"PF":>8}{"单笔均":>10}')
    S.log('-' * 96)
    for mm, b in itertools.product((1.5, 2.0, 2.5), (1.5, 2.0, 2.5)):
        sub = d3[(d3['calm3_maxmin'] <= mm) & (d3['burst3'] >= b)]
        st = stats(top5_by_day(sub, 'vol_ratio'))
        S.log(f'{"maxmin<=" + str(mm) + " & burst>=" + str(b) + " 量比排":<36}{st["n"]:>6}{st["win"]:>9.1f}{st["pnl"]:>12,.0f}{st["pf"]:>8.2f}{st["avg"]:>10,.0f}')
    S.log('')

    # ---------- D) 平静度分布 (供理解) ----------
    S.log('=' * 96)
    S.log('D) 平静度特征分布 (全量 1027 条)')
    S.log('=' * 96)
    for c in ('calm3_maxmin', 'burst3', 'calm5_maxmin', 'burst5', 'vol_ratio', 'ytd_vol_ratio'):
        if c in df.columns:
            s = df[c].dropna()
            S.log(f'{c:<18} 中位 {s.median():>7.2f}   25% {s.quantile(.25):>7.2f}   75% {s.quantile(.75):>7.2f}')


if __name__ == '__main__':
    main()

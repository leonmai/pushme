"""
止损/持有期敏感性分析 (stop_analysis)
=====================================
背景: 24 个月多月份验证显示, v7 策略「胜率 ~65% 但 PF < 1」——
      典型的高胜率、负期望形态: 多数单笔赚 +1% 止盈, 少数单笔暴跌 -20%~-33%。
      本脚本回答: 加止损 / 缩短持有期 能否把期望翻正。

做法: 复用 results_mv/all_signals.csv 的入场记录 (买入日 + 买入价),
      重新拉日K, 用「每日 高/低/开/收」逐日回放, 支持:
        - 止损位: 无 / -3% / -5% / -8% / -10%
        - 止盈位: +1% (v7 默认) / +3% / +5%
        - 持有期: 3 / 5 个交易日
      止损判定: 当日 最低价 <= 买价*(1-sl%) → 按 min(开盘, 止损价) 成交 (跳空保护)
      止盈判定: 当日 最高价 >= 买价*(1+tp%) → 按 max(开盘, 止盈价) 成交
      同日同时触发 → 保守按「先止损」处理
      末日强平: 按当日收盘

产出: results_mv/stop_grid.csv + 控制台结论

用法: python stop_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import monthly_validate as M

OUT = Path(__file__).parent / 'results_mv'


def load_dailies(codes: list, start: str, end: str) -> dict:
    """日K统一用 baostock (腾讯批量请求会触发 501 风控), 带磁盘缓存"""
    import screener_v2 as S
    syms = [f'{S.market_prefix(c)[:2]}.{S.code_format(c)}' for c in codes]
    dmap = M.bs_daily(syms, start, end, adjust='2', tag='止损分析')
    return {s.split('.')[1]: d for s, d in dmap.items()}


def replay(fut: pd.DataFrame, buy: float, sl: float, tp: float) -> dict:
    """fut: 买入日之后的 N 行日K (含 开/高/低/收). 返回 (pnl%, reason, days)"""
    for i, (_, r) in enumerate(fut.iterrows(), 1):
        o, h = float(r['开盘']), float(r['最高'])
        l, c = float(r['最低']), float(r['收盘'])
        stop_px = buy * (1 - sl / 100) if sl > 0 else -1
        tp_px = buy * (1 + tp / 100)
        # 跳空保护: 开盘已越过阈值则按开盘价成交
        if sl > 0 and o <= stop_px:
            return {'pnl_pct': (o / buy - 1) * 100, 'exit_reason': '止损', 'hold_days': i}
        if l <= stop_px and sl > 0:
            return {'pnl_pct': (stop_px / buy - 1) * 100, 'exit_reason': '止损', 'hold_days': i}
        if o >= tp_px:
            return {'pnl_pct': (o / buy - 1) * 100, 'exit_reason': '止盈', 'hold_days': i}
        if h >= tp_px:
            return {'pnl_pct': (tp_px / buy - 1) * 100, 'exit_reason': '止盈', 'hold_days': i}
        if i == len(fut):
            return {'pnl_pct': (c / buy - 1) * 100, 'exit_reason': '强平', 'hold_days': i}
    return {}


def stats(p: pd.Series, label: str, extra: dict) -> dict:
    if len(p) == 0:
        return {**extra, '组合': label, '交易数': 0}
    win, loss = p[p >= 0], p[p < 0]
    pf = win.sum() / abs(loss.sum()) if len(loss) and abs(loss.sum()) > 0 else np.inf
    return {**extra, '组合': label, '交易数': len(p),
            '胜率%': round((p >= 0).mean() * 100, 1),
            '平均收益%': round(p.mean(), 3),
            'PF': round(pf, 2) if np.isfinite(pf) else 999,
            '累计收益%': round(p.sum(), 1),
            '最差单笔%': round(p.min(), 1)}


def main():
    f = OUT / 'all_signals.csv'
    if not f.exists():
        M.log('缺少 all_signals.csv, 请先跑 monthly_validate.py')
        return
    A = pd.read_csv(f, dtype={'code': str})
    A['code'] = A['code'].astype(str).str.zfill(6)
    A['date'] = pd.to_datetime(A['date']).dt.date
    M.log(f'信号 {len(A)} 条, 月份 {A["yyyymm"].nunique()} 个')

    start = (pd.Timestamp(A['date'].min()) - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    end = (pd.Timestamp(A['date'].max()) + pd.Timedelta(days=40)).strftime('%Y-%m-%d')
    codes = sorted(A['code'].unique().tolist())
    M.log(f'拉日K {len(codes)} 只 ({start} ~ {end})')
    D = load_dailies(codes, start, end)
    M.log(f'日K 拿到 {len(D)} 只')

    # 每日 Top5 (按量比降序) — 与实盘一致
    A = A.sort_values(['date', 'vol_ratio'], ascending=[True, False])
    T5 = A.groupby('date', group_keys=False).head(5).reset_index(drop=True)
    M.log(f'Top5 组合 {len(T5)} 笔')

    rows = []
    for hold in (3, 5):
        for sl in (0, 3, 5, 8, 10):
            for tp in (1, 3, 5):
                res = []
                for _, r in T5.iterrows():
                    d = D.get(r['code'])
                    if d is None or d.empty:
                        continue
                    fut = d[d['日期'] > r['date']].head(hold)
                    if fut.empty:
                        continue
                    o = replay(fut, float(r['buy_price']), sl, tp)
                    if o:
                        res.append(o)
                if not res:
                    continue
                df = pd.DataFrame(res)
                rows.append(stats(df['pnl_pct'],
                                  f'止盈+{tp}%/{"不止损" if sl == 0 else f"止损-{sl}%"}',
                                  {'持有期': hold, '止损%': sl, '止盈%': tp}))
        # 大盘过滤版
        for sl in (0, 5, 8):
            g = T5[T5['market_ok'] == True]
            res = []
            for _, r in g.iterrows():
                d = D.get(r['code'])
                if d is None or d.empty:
                    continue
                fut = d[d['日期'] > r['date']].head(hold)
                if fut.empty:
                    continue
                o = replay(fut, float(r['buy_price']), sl, 1)
                if o:
                    res.append(o)
            if res:
                df = pd.DataFrame(res)
                rows.append(stats(df['pnl_pct'],
                                  f'止盈+1%/{("不止损" if sl == 0 else f"止损-{sl}%")} [大盘MA20]',
                                  {'持有期': hold, '止损%': sl, '止盈%': tp}))

    G = pd.DataFrame(rows)
    G.to_csv(OUT / 'stop_grid.csv', index=False, encoding='utf-8-sig')
    print('\n' + '=' * 90)
    print('止损 / 持有期 / 止盈 敏感性 (每日 Top5 组合)')
    print('=' * 90)
    print(G.to_string(index=False))

    best = G[G['交易数'] > 50].sort_values('PF', ascending=False).head(8)
    print('\nPF 最高的 8 个组合:')
    print(best.to_string(index=False))


if __name__ == '__main__':
    main()

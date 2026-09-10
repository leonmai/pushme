"""诊断今日扫描漏斗: 226 只候选 → 各条件淘汰了多少, 卡在哪一层"""
import sys
import json
import math
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import live_scout as L
import screener_v2 as S


def main():
    now = datetime.now()
    today = now.date()
    st = json.loads(L.STATE.read_text(encoding='utf-8')) if L.STATE.exists() else {}
    decl = st.get('decline', {}).get('data', {})

    snap = L.fetch_snapshot(600)
    snap = snap[~snap['code'].str.startswith(S.POOL_EXCLUDE_PREFIX)]
    snap = snap[~snap['name'].str.contains('ST|退', na=False)]
    snap = snap[(snap['turnover'] >= S.POOL_MIN_TURNOVER) & (snap['price'] >= 1.0) &
                (snap['pct'] >= -6) & (snap['pct'] <= 6)]
    snap = snap.sort_values('turnover', ascending=False).head(400).reset_index(drop=True)

    cands = [(r['code'], r['name'], decl[r['code']])
             for _, r in snap.iterrows() if r['code'] in decl and decl[r['code']] <= L.DECLINE_B]
    print(f"候选池 {len(snap)} → 符合近5日跌幅>1%: {len(cands)} 只\n")

    def work(item):
        code, name, dec = item
        try:
            df = L.fetch_15min_live(code)
            if df.empty:
                return None
            for c in ('open', 'close', 'volume'):
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df[~df['day'].apply(S.is_excluded_bar)].reset_index(drop=True)
            days = sorted(df['day'].dt.date.unique())
            if today not in days:
                return {'code': code, 'name': name, 'dec': dec, 'ytd': np.nan,
                        'nbar': 0, 'hitA': 0, 'hitAB': 0, 'best': None}
            prev = [d for d in days if d < today]
            dv = df.groupby(df['day'].dt.date)['volume'].sum()
            yv, tv = float(dv.get(prev[-1], 0)) if prev else 0.0, float(dv.get(today, 0))
            ytd = tv / yv if yv > 0 else np.nan

            bars = df[df['day'].dt.date == today].reset_index(drop=True)
            hitA = hitAB = 0
            best = None
            for i in range(1, len(bars)):
                cur, pv_bar = bars.iloc[i], bars.iloc[i - 1]
                op, cl = float(cur['open']), float(cur['close'])
                cv, pv = float(cur['volume']), float(pv_bar['volume'])
                if not all(map(math.isfinite, (op, cl, cv, pv))) or op <= 0 or pv <= 0:
                    continue
                chg = (cl - op) / op * 100
                vr = cv / pv
                okA = S.INTRADAY_PCT_MIN <= chg <= S.INTRADAY_PCT_MAX
                okB = vr >= S.VOL_MULT
                if okA:
                    hitA += 1
                if okA and okB:
                    hitAB += 1
                    if best is None or vr > best['vr']:
                        best = {'time': cur['day'].strftime('%H:%M'), 'chg': chg, 'vr': vr,
                                'close': cl, 'vol': cv, 'passed': ytd >= S.YESTERDAY_VOL_RATIO}
            return {'code': code, 'name': name, 'dec': dec, 'ytd': ytd,
                    'nbar': len(bars), 'hitA': hitA, 'hitAB': hitAB, 'best': best}
        except Exception:
            return None

    rows = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(work, c) for c in cands]
        for f in as_completed(futs):
            r = f.result()
            if r:
                rows.append(r)
    d = pd.DataFrame(rows)
    print(f"成功取到当日数据: {len(d)} 只\n")

    ysd = d['ytd'].dropna()
    print("=" * 76)
    print("【漏斗】今日各条件通过情况")
    print("=" * 76)
    print(f"  候选 (近5日跌幅>1%)              : {len(cands)}")
    print(f"  取到当日 15min 数据               : {len(d)}")
    print(f"  今/昨累计量 >= 0.9                : {(ysd >= 0.9).sum()}   <-- 关卡1")
    print(f"  有 bar 涨幅 0~2%                  : {(d['hitA'] > 0).sum()}   <-- 关卡2")
    print(f"  有 bar 同时满足 涨幅+量比>=2x     : {(d['hitAB'] > 0).sum()}   <-- 关卡3")
    print(f"  三关全过 (= 今日信号数)            : {len([r for r in rows if r['best'] and r['best']['passed']])}")

    print("\n" + "=" * 76)
    print("【今/昨累计量比分布】  (门槛 0.9)")
    print("=" * 76)
    print(f"  中位数 {ysd.median():.2f}   均值 {ysd.mean():.2f}   最大 {ysd.max():.2f}")
    for th in (0.5, 0.7, 0.8, 0.9, 1.0):
        print(f"  >= {th:.1f} 的股票数: {(ysd >= th).sum()}")

    near = [r for r in rows if r['best']]
    print("\n" + "=" * 76)
    print(f"【最接近信号的个股】满足 涨幅0~2% + 量比>=2x 的共 {len(near)} 只, 按量比排序")
    print("=" * 76)
    near.sort(key=lambda r: r['best']['vr'], reverse=True)
    print(f"{'名称':<10}{'代码':<11}{'bar':<7}{'近5日跌':>8}{'当根涨':>8}{'量比':>8}{'今/昨':>8}  状态")
    for r in near[:20]:
        b = r['best']
        flag = "已通过" if b['passed'] else f"卡今昨({r['ytd']:.2f})"
        print(f"{r['name'][:9]:<10}{r['code']:<11}{b['time']:<7}{r['dec']:>7.1f}%"
              f"{b['chg']:>7.2f}%{b['vr']:>7.2f}x{r['ytd']:>8.2f}  {flag}")


if __name__ == '__main__':
    main()

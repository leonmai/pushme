import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
import screener_v2 as S

code = '000725'
print('=' * 88)
print('京东方A (sz000725)  2026-08-04  15分钟明细')
print('=' * 88)

# 1) 15min 数据
df = S.fetch_15min_bs(code, 2026, 8)
if df is None or df.empty:
    print('无数据')
    sys.exit()
df['day'] = pd.to_datetime(df['day'])
for c in ('open', 'high', 'low', 'close', 'volume'):
    df[c] = pd.to_numeric(df[c], errors='coerce')

d = df[df['day'].dt.date.astype(str) == '2026-08-04'].reset_index(drop=True)
print(f'\n当日共 {len(d)} 根 bar\n')
print(f'{"时间":<8}{"开":>9}{"收":>9}{"bar内涨幅":>11}{"成交量(万手)":>14}{"vs前根":>9}{"vs当日均量":>12}')
print('-' * 88)

avg_vol = d['volume'].mean()
prev = None
for _, r in d.iterrows():
    t = r['day'].strftime('%H:%M')
    op, cl, v = float(r['open']), float(r['close']), float(r['volume'])
    chg = (cl - op) / op * 100 if op > 0 else 0
    vs_prev = f'{v/prev:.2f}x' if prev and prev > 0 else '-'
    vs_avg = f'{v/avg_vol:.2f}x'
    mark = '  <<< 信号' if t == '13:30' else ''
    print(f'{t:<8}{op:>9.3f}{cl:>9.3f}{chg:>10.2f}%{v/1e4:>13.0f}{vs_prev:>10}{vs_avg:>12}{mark}')
    prev = v

# 2) 近5日跌幅 (不含当天盘中)
start = '2026-06-20'
end = '2026-08-04'
ddf = S.fetch_daily(code, start, end)
if not ddf.empty:
    ddf['日期'] = pd.to_datetime(ddf['日期'])
    recent = ddf[ddf['日期'] <= '2026-08-04'].tail(6)
    print('\n' + '=' * 88)
    print('近 6 个交易日日K (用于算近5日累计跌幅, 不含 08-04 盘中)')
    print('=' * 88)
    print(f'{"日期":<13}{"收盘":>9}{"日涨跌":>10}{"成交量(万手)":>14}')
    print('-' * 88)
    closes = recent['收盘'].astype(float).tolist()
    vols = recent['成交量'].astype(float).tolist() if '成交量' in recent.columns else [0]*len(recent)
    for i, (_, r) in enumerate(recent.iterrows()):
        c = float(r['收盘'])
        pct = (c / closes[i-1] - 1) * 100 if i > 0 else 0
        v = float(r['成交量']) / 1e4 if '成交量' in r else 0
        tag = '  <- 信号日' if str(r['日期'])[:10] == '2026-08-04' else ''
        print(f'{str(r["日期"])[:10]:<13}{c:>9.2f}{pct:>9.2f}%{v:>13.0f}{tag}')
    # 近5日累计跌幅: 用信号日之前的5个交易日 (t-5 -> t-1)
    if len(closes) >= 6:
        cum = (closes[4] / closes[0] - 1) * 100   # t-5 到 t-1
        print('-' * 88)
        print(f'近5日累计跌幅 (07-28收盘 {closes[0]:.2f} -> 08-03收盘 {closes[4]:.2f}): {cum:+.2f}%')

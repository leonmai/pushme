"""京东方A 8/4 平静蓄势形态: 当前算法 vs 价格维度对照"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import live_scout as L
import screener_v2 as S

code = '000725'
df = L.fetch_15min_live(code)
print(f"新浪取到 {len(df)} 根, 范围 {df['day'].min()} ~ {df['day'].max()}")
if '2026-08-04' not in set(df['day'].dt.date.astype(str)):
    print("  → 8/4 不在新浪覆盖范围, 改用 baostock 历史 15min")
    df = S.fetch_15min_bs(code, 2026, 8)
    df['day'] = pd.to_datetime(df['day'])
    for c in ('open', 'high', 'low', 'close', 'volume'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    print(f"  baostock 取到 {len(df)} 根, 范围 {df['day'].min()} ~ {df['day'].max()}")

d = df[df['day'].dt.date.astype(str) == '2026-08-04'].copy()
d = d[~d['day'].apply(S.is_excluded_bar)].reset_index(drop=True)
print(f"8/4 有效 bar: {len(d)} 根 (已剔除 11:30/13:15/15:00)\n")

print(f"{'时间':<8}{'开':>7}{'收':>7}{'当根涨跌':>10}{'成交量(万)':>12}{'量比(vs前根)':>14}")
prev_v = None
for _, r in d.iterrows():
    chg = (r['close'] - r['open']) / r['open'] * 100
    vr = (r['volume'] / prev_v) if prev_v else float('nan')
    mark = '  <== 信号根' if r['day'].strftime('%H:%M') == '13:45' else ''
    print(f"{r['day'].strftime('%H:%M'):<8}{r['open']:>7.2f}{r['close']:>7.2f}"
          f"{chg:>9.2f}%{r['volume']/1e4:>12.0f}{vr:>14.2f}{mark}")
    prev_v = r['volume']

sig_idx = d.index[d['day'].dt.strftime('%H:%M') == '13:45'][0]
cv = float(d.iloc[sig_idx]['volume'])
vs = [float(d.iloc[i]['volume']) for i in range(sig_idx - 3, sig_idx)]
calm3 = max(vs) / min(vs)
burst3 = cv / (sum(vs) / len(vs))
print(f"\n--- 当前算法 (量能维度) ---")
print(f"  前3根量: {[round(v/1e4) for v in vs]} 万手")
print(f"  平静度 calm3 = max/min = {calm3:.2f}   (越小越均匀, 门槛 <=1.5)")
print(f"  爆发   burst3 = 信号根/前3根均量 = {cv/1e4:.0f}/{sum(vs)/len(vs)/1e4:.0f} = {burst3:.2f}  (门槛 >=2)")
print(f"  → {'★ 平静蓄势形态' if calm3 <= 1.5 and burst3 >= 2 else '未达形态标准'}")

print(f"\n--- 你提的口径 (价格维度: 连续 |涨跌|<1% 的根数) ---")
n = 0
for i in range(sig_idx - 1, -1, -1):
    chg = (d.iloc[i]['close'] - d.iloc[i]['open']) / d.iloc[i]['open'] * 100
    if abs(chg) < 1.0:
        n += 1
    else:
        print(f"  第 {n+1} 根前 ({d.iloc[i]['day'].strftime('%H:%M')}) 涨跌 {chg:+.2f}% 超过 1%, 中断")
        break
print(f"  平静根数 = {n} 根 ({n*15} 分钟)")
seg = d.iloc[sig_idx - n:sig_idx]
amp = (seg['high'].max() - seg['low'].min()) / seg['low'].min() * 100
print(f"  该区间整体振幅 = {amp:.2f}%")

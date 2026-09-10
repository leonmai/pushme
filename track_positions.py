"""持仓跟踪: 记录每日买入的信号股, 按规则判定 T+1~T+5 出场

规则:
  - 买入价 = 信号根 15min 收盘价
  - T+1 起 5 个交易日内, 日K收盘 >= 买入价 x 1.05 -> 止盈卖出   (v8: 由 1.01 上调)
  - 第 5 个交易日收盘仍未止盈 -> 强平
  - 不加止损 (回测验证: 加止损会砍掉先挖坑再反弹的票, PF 反而下降)
  - 不加止损
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import live_scout as L
import push_notify as PN

OUT = Path(__file__).parent / 'results_live'
POS = OUT / 'positions.json'
# 2026-09-09 (v8): 由 1.0 上调至 5.0。
# 依据: 24 个月 / 11469 条信号回测, 止盈 +1% 时 PF 1.13、平均单笔 +0.15%;
#       放宽到 +5% 后 PF 1.35、平均单笔 +0.71% —— +1% 出场过早, 把利润让出去了。
#       同时验证过: 加止损(-3%/-5%)会让 PF 降到 0.87/0.99, 故保持不止损。
TP_PCT = 5.0
HOLD_DAYS = 5
CAPITAL = 10_000


def load():
    if POS.exists():
        return json.loads(POS.read_text(encoding='utf-8'))
    return {'positions': [], 'closed': []}


def save(d):
    OUT.mkdir(exist_ok=True)
    POS.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')


def fetch_daily(code: str) -> pd.DataFrame:
    """日K (前复权), 多源降级。

    主源: live_scout.fetch_daily_qfq —— 纯 requests 直连腾讯, **云端无 akshare 也能跑**。
    备用: akshare stock_zh_a_daily (本机若装了就用)。
    注意: 之前只依赖 akshare, 在没装 akshare 的云端环境会永远返回空,
    导致持仓永远无法判定出场 —— 这是云端部署必须修的点。"""
    sym = L.S.market_prefix(code)
    # 1) 主源: 纯 requests 腾讯日K (云端友好)
    try:
        df = L.fetch_daily_qfq(code)
        if df is not None and not df.empty:
            keep = [c for c in ('date', 'open', 'high', 'low', 'close') if c in df.columns]
            return df[keep].copy()
    except Exception:
        pass
    # 2) 备用: akshare (需本地安装)
    try:
        import akshare as ak
        df = ak.stock_zh_a_daily(symbol=sym, adjust='qfq')
        if df is not None and not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            return df[['date', 'open', 'high', 'low', 'close']].copy()
    except Exception:
        pass
    return pd.DataFrame()


def init_from_signals(date_str: str, max_positions: int = 3):
    """从当日信号CSV建立持仓 (按同期放量 ytd_same 降序取前 N 只, 与 v9 排序一致)"""
    f = OUT / f'signals_{date_str}.csv'
    if not f.exists():
        print(f'无信号文件 {f}')
        return []
    d = pd.read_csv(f)
    d['code'] = d['code'].astype(str).str.zfill(6)
    d = d.drop_duplicates(subset=['code', 'bar_time'])
    # v9 排序键: 优先同期放量 ytd_same, 回退瞬时量比 vol_ratio
    sort_key = 'ytd_same' if 'ytd_same' in d.columns else 'vol_ratio'
    d = d.sort_values(sort_key, ascending=False).head(max_positions)
    # 名称兜底 (CSV 里可能是代码)
    nm = L.fetch_names(d['code'].tolist())
    d['name'] = [nm.get(c, n) if str(n) == str(c) else n
                 for c, n in zip(d['code'], d['name'])]
    out = []
    for _, r in d.iterrows():
        out.append({
            'code': r['code'], 'name': str(r['name']), 'buy_date': date_str,
            'buy_price': float(r['close']), 'buy_bar': str(r['bar_time'])[11:16],
            'vol_ratio': float(r['vol_ratio']), 'calm': float(r['calm3_maxmin']),
            'burst': float(r['burst3']), 'grade': str(r['grade']),
            'decline_5d': float(r['decline_5d_pct']),
            'hold_day': 0,
            'shares': max(1, int(CAPITAL / float(r['close']) / 100)) * 100,
        })
    return out


def fetch_realtime(codes: list) -> dict:
    """个股实时价 (新浪 hq, 纯 requests)"""
    out = {}
    syms = [L.S.market_prefix(c) for c in codes]
    for i in range(0, len(syms), 60):
        try:
            r = L.requests.get('http://hq.sinajs.cn/list=' + ','.join(syms[i:i + 60]),
                               headers=L.HDR, timeout=15)
            r.encoding = 'gbk'
            for line in r.text.strip().split('\n'):
                m = L.re.search(r'hq_str_([a-z]{2}\d+)="([^"]+)"', line)
                if not m:
                    continue
                p = m.group(2).split(',')
                if len(p) < 4:
                    continue
                code = m.group(1)[2:]
                try:
                    out[code] = (p[0], float(p[3]))
                except ValueError:
                    pass
        except Exception:
            pass
    return out


def update(data: dict) -> list:
    """检查未出场持仓, 返回本次新出场的列表"""
    closed_now = []
    for p in data['positions']:
        df = fetch_daily(p['code'])
        if df.empty:
            continue
        df = df[df['date'] > pd.Timestamp(p['buy_date'])]
        if df.empty:
            continue
        target = p['buy_price'] * (1 + TP_PCT / 100)
        exited = False
        for i, (_, r) in enumerate(df.iterrows(), start=1):
            cl = float(r['close'])
            if cl >= target:
                p.update(exit_date=str(r['date'].date()), exit_price=cl,
                         exit_reason='止盈', hold_day=i)
                exited = True
                break
            if i >= HOLD_DAYS:
                p.update(exit_date=str(r['date'].date()), exit_price=cl,
                         exit_reason='强平', hold_day=i)
                exited = True
                break
        if exited:
            p['pnl_pct'] = (p['exit_price'] - p['buy_price']) / p['buy_price'] * 100
            p['pnl'] = (p['exit_price'] - p['buy_price']) * p['shares']
            closed_now.append(p)
    if closed_now:
        codes = {p['code'] for p in closed_now}
        data['positions'] = [p for p in data['positions'] if p['code'] not in codes]
        data['closed'].extend(closed_now)
    return closed_now


def report(data: dict, closed_now: list):
    print('=' * 92)
    print('  持仓跟踪')
    print('=' * 92)
    if data['positions']:
        rt = fetch_realtime([p['code'] for p in data['positions']])
        print(f'\n【持仓中 {len(data["positions"])} 只】')
        print(f'{"名称":<10}{"代码":<8}{"买入日":<12}{"买入价":>9}{"已持有":>7}'
              f'{"量比":>7}{"级别":>5}  最新')
        for p in data['positions']:
            df = fetch_daily(p['code'])
            last = ''
            d_after = df[df['date'] > pd.Timestamp(p['buy_date'])] if not df.empty else df
            if not d_after.empty:
                r = d_after.iloc[-1]
                chg = (float(r['close']) - p['buy_price']) / p['buy_price'] * 100
                tag = '收' if str(r['date'].date()) == str(pd.Timestamp.now().date()) else '收'
                last = f"{str(r['date'].date())} {tag}{float(r['close']):.2f} ({chg:+.2f}%)"
            elif p['code'] in rt and rt[p['code']][1] > 0:
                px = rt[p['code']][1]
                chg = (px - p['buy_price']) / p['buy_price'] * 100
                last = f"实时 {px:.2f} ({chg:+.2f}%)  [日K未更新]"
            if p['name'] == p['code'] and p['code'] in rt:
                p['name'] = rt[p['code']][0]
            print(f"{p['name']:<10}{p['code']:<8}{p['buy_date']:<12}"
                  f"{p['buy_price']:>9.2f}{p['hold_day']:>7}"
                  f"{p['vol_ratio']:>7.2f}{p['grade']:>5}  {last}")
    else:
        print('\n【当前无持仓】')

    if closed_now:
        print(f'\n【本次出场 {len(closed_now)} 只】')
        for p in closed_now:
            print(f"  {p['name']:<10} {p['code']}  {p['exit_reason']} "
                  f"T+{p['hold_day']}  {p['buy_price']:.2f} → {p['exit_price']:.2f}  "
                  f"{p['pnl_pct']:+.2f}%  {p['pnl']:+,.0f}元")

    cl = data['closed']
    if cl:
        n = len(cl)
        win = sum(1 for p in cl if p['pnl'] > 0)
        tot = sum(p['pnl'] for p in cl)
        gp = sum(p['pnl'] for p in cl if p['pnl'] > 0)
        gl = -sum(p['pnl'] for p in cl if p['pnl'] < 0)
        pf = gp / gl if gl else float('inf')
        print(f'\n【累计已平仓 {n} 笔】胜率 {win / n * 100:.1f}%  '
              f'合计 {tot:+,.0f}元  PF {pf:.2f}')
        for p in cl[-10:]:
            print(f"  {p['buy_date']} {p['name']:<10} {p['code']} "
                  f"{p['exit_reason']} T+{p['hold_day']} {p['pnl_pct']:+.2f}% "
                  f"{p['pnl']:+,.0f}元")


def build_track_html(data: dict, closed_now: list) -> str:
    """把持仓跟踪结果渲染成 HTML, 供 PushPlus 推送。"""
    _now = L.now_cst()            # 统一用北京时间, 云端 runner 是 UTC 也不会错
    rows = []
    for p in data['positions']:
        df = fetch_daily(p['code'])
        last = '—'
        d_after = df[df['date'] > pd.Timestamp(p['buy_date'])] if not df.empty else df
        if not d_after.empty:
            r = d_after.iloc[-1]
            chg = (float(r['close']) - p['buy_price']) / p['buy_price'] * 100
            last = f"{str(r['date'].date())} 收 {float(r['close']):.2f} ({chg:+.2f}%)"
        rt = fetch_realtime([p['code']])
        if p['code'] in rt and rt[p['code']][1] > 0:
            px = rt[p['code']][1]
            chg = (px - p['buy_price']) / p['buy_price'] * 100
            last = f"实时 {px:.2f} ({chg:+.2f}%)"
        target = p['buy_price'] * (1 + TP_PCT / 100)
        rows.append(
            f"<tr><td><b>{p['name']}</b><br><span style='color:#999;font-size:12px'>"
            f"{p['code']}</span></td><td>{p['buy_date']} {p['buy_bar']}</td>"
            f"<td class='num'>{p['buy_price']:.2f}</td>"
            f"<td class='num'>+{TP_PCT:.0f}%→{target:.2f}</td>"
            f"<td class='num'>{p['grade']}</td><td>{last}</td></tr>")
    pos_html = ''.join(rows) if rows else "<tr><td colspan='6' style='color:#999'>当前无持仓</td></tr>"

    closed_rows = []
    for p in closed_now:
        color = '#0a8f4e' if p['pnl'] > 0 else '#c0392b'
        closed_rows.append(
            f"<tr><td><b>{p['name']}</b> {p['code']}</td>"
            f"<td>{p['exit_reason']} T+{p['hold_day']}</td>"
            f"<td class='num'>{p['buy_price']:.2f}→{p['exit_price']:.2f}</td>"
            f"<td class='num' style='color:{color};font-weight:700'>{p['pnl_pct']:+.2f}%</td>"
            f"<td class='num' style='color:{color}'>{p['pnl']:+,.0f}元</td></tr>")
    closed_html = ''.join(closed_rows) if closed_rows else \
        "<tr><td colspan='5' style='color:#999'>本次无出场</td></tr>"

    cl = data['closed']
    if cl:
        n = len(cl)
        win = sum(1 for p in cl if p['pnl'] > 0)
        tot = sum(p['pnl'] for p in cl)
        gp = sum(p['pnl'] for p in cl if p['pnl'] > 0)
        gl = -sum(p['pnl'] for p in cl if p['pnl'] < 0)
        pf = gp / gl if gl else float('inf')
        stat = (f"累计已平仓 <b>{n}</b> 笔 · 胜率 <b>{win / n * 100:.1f}%</b> · "
                f"合计 <b>{tot:+,.0f}</b> 元 · PF <b>{pf:.2f}</b>")
    else:
        stat = "累计已平仓 0 笔"

    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>持仓跟踪 {_now.strftime('%Y-%m-%d')}</title>
<style>body{{font-family:"Microsoft YaHei",sans-serif;background:#f5f6f8;color:#1f2430;margin:0;padding:20px}}
.wrap{{max-width:820px;margin:0 auto}} h1{{font-size:18px}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px;margin-bottom:16px}}
th,td{{border:1px solid #e6e8ef;padding:6px 8px;text-align:left}} th{{background:#f0f2f7}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
.stat{{background:#fff;border:1px solid #e6e8ef;border-radius:8px;padding:10px 14px;font-size:14px}}
.h{{font-size:15px;font-weight:700;margin:14px 0 6px}}</style></head>
<body><div class="wrap"><h1>持仓跟踪 · {_now.strftime('%Y-%m-%d %H:%M')}</h1>
<div class="stat">{stat}</div>
<div class="h">持仓中 {len(data['positions'])} 只</div>
<table><tr><th>名称/代码</th><th>买入(日/bar)</th><th>买入价</th><th>止盈目标</th><th>级别</th><th>最新</th></tr>
{pos_html}</table>
<div class="h">本次出场 {len(closed_now)} 只</div>
<table><tr><th>名称/代码</th><th>出场</th><th>买→卖</th><th>收益率</th><th>盈亏</th></tr>
{closed_html}</table>
<div style="font-size:12px;color:#8a93a6;margin-top:8px">
规则: 信号根收盘价买入, 每只约 1 万元; T+1 起 5 个交易日日K收盘 ≥ 买入价×1.05 止盈, 第 5 日强平, 不加止损。<br>
本推送为「若买入」的纸上模拟, 非真实成交; 实盘请在券商 APP 挂 +5% 条件单。</div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', help='从指定日期的信号CSV建立持仓, 如 2026-09-07')
    ap.add_argument('--max-positions', type=int, default=3)
    args = ap.parse_args()

    data = load()
    if args.init:
        new = init_from_signals(args.init, args.max_positions)
        have = {p['code'] for p in data['positions']}
        new = [p for p in new if p['code'] not in have]
        data['positions'].extend(new)
        save(data)
        print(f'建立持仓 {len(new)} 只:')
        for p in new:
            print(f"  {p['name']:<10} {p['code']}  {p['buy_bar']}  "
                  f"买入价 {p['buy_price']:.2f}  量比 {p['vol_ratio']:.2f}×  "
                  f"{p['shares']}股")
        # 建仓后也推送一条, 让用户知道今日模拟持仓已建立 (仅在有新建时)
        if new:
            try:
                PN.push_html(f"模拟建仓 {args.init} · {len(new)} 只",
                            build_track_html(data, []))
            except Exception as e:
                print(f'推送异常(不影响建仓): {e}')
        return
    closed_now = update(data)
    save(data)
    report(data, closed_now)
    # 微信推送持仓跟踪结果
    try:
        _today = L.now_cst().date().strftime('%Y-%m-%d')
        PN.push_html(f"持仓跟踪 {_today}", build_track_html(data, closed_now))
    except Exception as e:
        print(f'推送异常(不影响跟踪): {e}')


if __name__ == '__main__':
    main()

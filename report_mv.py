"""
多月份验证报告 (report_mv)
==========================
读 results_mv/monthly_summary.csv + total_summary.csv, 生成 results_mv/report.html
纯内联 SVG 绘图, 离线可看, light theme.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / 'results_mv'

CSS = """
*{box-sizing:border-box}
body{font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;background:#f7f8fa;
color:#1a1d24;margin:0;padding:28px;line-height:1.6}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:32px 0 12px;padding-left:10px;border-left:4px solid #2f6fed}
.sub{color:#6b7280;font-size:13px;margin-bottom:20px}
.cards{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 18px;min-width:150px;flex:1}
.card .k{font-size:12px;color:#6b7280}
.card .v{font-size:24px;font-weight:600;margin-top:2px}
.pos{color:#d92b2b}.neg{color:#0f9960}
table{border-collapse:collapse;width:100%;background:#fff;font-size:13px;
border:1px solid #e5e7eb;border-radius:8px;overflow:hidden}
th{background:#f1f3f7;padding:8px 10px;text-align:right;font-weight:600;color:#374151}
th:first-child,td:first-child{text-align:left}
td{padding:7px 10px;text-align:right;border-top:1px solid #f0f1f4}
tr:hover td{background:#fafbfc}
.note{background:#fff8e6;border:1px solid #f5d98a;border-radius:8px;padding:12px 16px;margin:14px 0;font-size:14px}
.ok{background:#eefbf3;border-color:#9ad9b8}
.bad{background:#fdeeee;border-color:#f0aaaa}
ul{margin:8px 0 8px 20px;padding:0}
li{margin:4px 0}
"""

MONTH_CN = {1: '1月', 2: '2月', 3: '3月', 4: '4月', 5: '5月', 6: '6月', 7: '7月',
            8: '8月', 9: '9月', 10: '10月', 11: '11月', 12: '12月'}


def fmt_m(s):
    s = str(s)
    return f"{s[:4]}年{int(s[4:6])}月"


def bar_chart(labels, values, width=1120, height=260, title=''):
    """内联 SVG 柱状图 (红涨绿跌)"""
    n = len(values)
    if n == 0:
        return ''
    pad_l, pad_r, pad_t, pad_b = 46, 12, 24, 46
    iw = width - pad_l - pad_r
    ih = height - pad_t - pad_b
    vmax = max(max(values), 0)
    vmin = min(min(values), 0)
    span = (vmax - vmin) or 1
    zero_y = pad_t + ih * (vmax / span)
    bw = iw / n * 0.68
    gap = iw / n
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'style="background:#fff;border:1px solid #e5e7eb;border-radius:8px">']
    if title:
        parts.append(f'<text x="{pad_l}" y="16" font-size="13" fill="#374151">{title}</text>')
    parts.append(f'<line x1="{pad_l}" y1="{zero_y}" x2="{width - pad_r}" y2="{zero_y}" '
                 f'stroke="#9aa1ad" stroke-width="1"/>')
    for i, v in enumerate(values):
        x = pad_l + i * gap + (gap - bw) / 2
        h = abs(v) / span * ih
        y = zero_y - h if v >= 0 else zero_y
        color = '#d92b2b' if v >= 0 else '#0f9960'
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(h,1):.1f}" '
                     f'fill="{color}" opacity="0.85" rx="2"/>')
        if v != 0:
            ty = y - 4 if v >= 0 else y + h + 12
            parts.append(f'<text x="{x + bw / 2:.1f}" y="{ty:.1f}" font-size="9" '
                         f'fill="{color}" text-anchor="middle">{v:.0f}</text>')
        if n <= 26:
            lb = labels[i]
            parts.append(f'<text x="{x + bw / 2:.1f}" y="{height - pad_b + 14:.1f}" font-size="9" '
                         f'fill="#6b7280" text-anchor="middle">{lb}</text>')
    parts.append(f'<text x="{pad_l - 6}" y="{pad_t + 10}" font-size="10" fill="#9aa1ad" '
                 f'text-anchor="end">{vmax:.0f}</text>')
    parts.append(f'<text x="{pad_l - 6}" y="{pad_t + ih:.0f}" font-size="10" fill="#9aa1ad" '
                 f'text-anchor="end">{vmin:.0f}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def main():
    ms = pd.read_csv(OUT / 'monthly_summary.csv')
    ts = pd.read_csv(OUT / 'total_summary.csv')
    A = pd.read_csv(OUT / 'all_signals.csv', dtype={'code': str})

    t5 = ms[ms['分组'] == 'Top5'].sort_values('月份')
    allm = ms[ms['分组'] == '全信号'].sort_values('月份')
    mkt = ms[ms['分组'] == 'Top5+大盘'].sort_values('月份')

    def row_of(df, label):
        r = ts[ts['分组'] == label]
        return r.iloc[0] if len(r) else None

    r_top5 = row_of(ts, '每日Top5')
    r_all = row_of(ts, '全信号')
    r_mkt = row_of(ts, 'Top5 + 大盘MA20')
    r_a = row_of(ts, 'A级 Top5')
    r_b = row_of(ts, 'B级 Top5')

    win_months = int((t5['累计收益%'] > 0).sum())
    tot_months = len(t5)

    # ---- 结论 ----
    pf = float(r_top5['PF']) if r_top5 is not None else 0
    avg = float(r_top5['平均收益%']) if r_top5 is not None else 0
    verdict_cls = 'ok' if (pf >= 1.2 and avg > 0) else ('bad' if pf < 1 else '')
    if pf >= 1.5 and avg > 0.3:
        vtxt = '多月份验证通过：策略在样本外依然稳定为正期望。'
    elif pf >= 1.1 and avg > 0:
        vtxt = '策略整体有效，但优势偏薄——扣掉手续费/滑点后可能接近打平，需加止损保护。'
    elif pf >= 0.95:
        vtxt = '策略接近零期望：胜率高但被少数大亏吃掉，实盘不建议裸用。'
    else:
        vtxt = '策略在多月份样本中为负期望，此前三月回测的正收益很可能是样本选择造成的假象。'
    mkt_pf = float(r_mkt['PF']) if r_mkt is not None and pd.notna(r_mkt['PF']) else 0
    mkt_n = int(r_mkt['交易数']) if r_mkt is not None else 0

    H = [f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
         '<title>A股 v7 策略 多月份验证报告</title>', f'<style>{CSS}</style></head><body><div class="wrap">',
         '<h1>A股 15min 量价异动 v7 — 多月份有效性验证</h1>',
         f'<div class="sub">样本：日均成交额前 300 只 · 24 个月（{fmt_m(t5["月份"].iloc[0])} ~ '
         f'{fmt_m(t5["月份"].iloc[-1])}） · 每日按量比取 Top5 · 每笔 1 万元等权 · '
         f'出场：T+1 起 5 日内 +1% 止盈，否则第 5 日强平（不止损）</div>']

    H.append(f'<div class="note {verdict_cls}"><b>结论：{vtxt}</b><br>'
             f'全样本每日 Top5 组合：PF <b>{pf:.2f}</b>、平均单笔 <b>{avg:+.2f}%</b>、'
             f'胜率 <b>{float(r_top5["胜率%"]):.1f}%</b>、'
             f'{tot_months} 个月中 <b>{win_months} 个月盈利</b>。'
             f'叠加「上证 &gt; MA20」后：PF {mkt_pf:.2f}（{mkt_n} 笔）。</div>')

    # 指标卡
    def card(k, v, cls=''):
        return f'<div class="card"><div class="k">{k}</div><div class="v {cls}">{v}</div></div>'

    H.append('<div class="cards">')
    H.append(card('样本外 PF', f'{pf:.2f}', 'pos' if pf >= 1 else 'neg'))
    H.append(card('平均单笔', f'{avg:+.2f}%', 'pos' if avg > 0 else 'neg'))
    H.append(card('胜率', f'{float(r_top5["胜率%"]):.1f}%'))
    H.append(card('总交易', f'{int(r_top5["交易数"])}'))
    H.append(card('盈利月份', f'{win_months}/{tot_months}'))
    H.append(card('最差单笔', f'{float(r_top5["最差单笔%"]):.1f}%', 'neg'))
    H.append('</div>')

    # 月度柱状图
    H.append('<h2>每月净收益（每日 Top5 组合，%）—— 红涨绿跌</h2>')
    H.append(bar_chart([fmt_m(x)[2:] for x in t5['月份']],
                       [float(v) for v in t5['累计收益%']],
                       title='每月累计收益%（等权叠加，非复利）'))
    H.append('<div class="sub">注：柱高为该月所有交易盈亏的简单相加，未复利；'
             '每月交易笔数不同，横向比较看方向而非绝对值。</div>')

    # 月度明细
    H.append('<h2>逐月明细</h2>')
    cols = ['月份', '交易数', '胜率%', '止盈率%', '平均收益%', '中位收益%', 'PF',
            '盈亏比', '累计收益%', '最差单笔%', '最大连亏']
    H.append('<table><tr>' + ''.join(f'<th>{c}</th>' for c in ['月份', '分组'] + cols[1:]) + '</tr>')
    for _, r in ms.sort_values(['月份', '分组']).iterrows():
        cls = 'pos' if (pd.notna(r['累计收益%']) and r['累计收益%'] > 0) else (
            'neg' if pd.notna(r['累计收益%']) and r['累计收益%'] < 0 else '')
        cells = ''.join(
            f'<td class="{cls if c == "累计收益%" else ""}">'
            f'{"-" if pd.isna(r[c]) else (fmt_m(r[c]) if c == "月份" else r[c])}</td>'
            for c in cols[1:])
        H.append(f'<tr><td>{fmt_m(r["月份"])}</td><td>{r["分组"]}</td>{cells}</tr>')
    H.append('</table>')

    # 总体分组对比
    H.append('<h2>全样本分组对比</h2>')
    H.append('<table><tr><th>分组</th><th>交易数</th><th>胜率%</th><th>止盈率%</th>'
             '<th>平均收益%</th><th>中位收益%</th><th>PF</th><th>盈亏比</th>'
             '<th>累计收益%</th><th>最差单笔%</th></tr>')
    for _, r in ts.iterrows():
        cls = 'pos' if (pd.notna(r['PF']) and r['PF'] >= 1) else 'neg'
        H.append(f'<tr><td>{r["分组"]}</td><td>{int(r["交易数"])}</td><td>{r["胜率%"]}</td>'
                 f'<td>{r["止盈率%"]}</td><td>{r["平均收益%"]}</td><td>{r["中位收益%"]}</td>'
                 f'<td class="{cls}">{r["PF"]}</td><td>{r["盈亏比"]}</td>'
                 f'<td>{r["累计收益%"]}</td><td class="neg">{r["最差单笔%"]}</td></tr>')
    H.append('</table>')

    # 尾部风险
    p = A['pnl_pct']
    big_loss = A[A['pnl_pct'] <= -8]
    H.append('<h2>尾部风险（为什么高胜率仍可能亏钱）</h2>')
    H.append('<div class="cards">')
    H.append(card('P(亏损 > 8%)', f'{(p <= -8).mean() * 100:.1f}%', 'neg'))
    H.append(card('P(亏损 > 15%)', f'{(p <= -15).mean() * 100:.1f}%', 'neg'))
    H.append(card('亏损笔平均', f'{p[p < 0].mean():.2f}%' if (p < 0).any() else '-', 'neg'))
    H.append(card('盈利笔平均', f'{p[p >= 0].mean():.2f}%' if (p >= 0).any() else '-', 'pos'))
    H.append('</div>')
    H.append(f'<div class="note">止盈固定在 +1%，盈利端被封顶；亏损端完全裸露，'
             f'最差单笔 <b>{p.min():.1f}%</b>。'
             f'这种「赚小钱、亏大钱」的结构，胜率再高也容易被尾部拖垮 —— '
             f'这正是本策略在不同月份表现分化的根源。</div>')

    if not big_loss.empty:
        bl = big_loss[['code', 'name', 'date', 'vol_ratio', 'decline_5d_pct',
                       'pnl_pct', 'exit_reason']].head(12)
        H.append('<table><tr><th>代码</th><th>名称</th><th>买入日</th><th>量比</th>'
                 '<th>近5日跌%</th><th>收益%</th><th>出场</th></tr>')
        for _, r in bl.iterrows():
            H.append(f'<tr><td>{r["code"]}</td><td>{r["name"]}</td><td>{r["date"]}</td>'
                     f'<td>{r["vol_ratio"]:.2f}</td><td>{r["decline_5d_pct"]:.1f}</td>'
                     f'<td class="neg">{r["pnl_pct"]:.1f}</td><td>{r["exit_reason"]}</td></tr>')
        H.append('</table>')

    # 止盈/止损敏感性
    sg = OUT / 'stop_grid.csv'
    if sg.exists():
        G = pd.read_csv(sg)
        G5 = G[(G['持有期'] == 5) & (G['止损%'] == 0) & (~G['组合'].str.contains('大盘'))]
        Gs = G[(G['持有期'] == 5) & (G['止盈%'] == 1) & (~G['组合'].str.contains('大盘'))]
        H.append('<h2>出场参数敏感性（同一个信号池，只改出场规则）</h2>')
        H.append('<h3>止盈目标（不止损，持有 5 日）</h3>')
        H.append('<table><tr><th>止盈</th><th>胜率%</th><th>平均收益%</th><th>PF</th>'
                 '<th>累计收益%</th><th>最差单笔%</th></tr>')
        for _, r in G5.sort_values('止盈%').iterrows():
            cls = 'pos' if r['PF'] >= 1.2 else ''
            H.append(f'<tr><td>+{int(r["止盈%"])}%</td><td>{r["胜率%"]}</td>'
                     f'<td>{r["平均收益%"]}</td><td class="{cls}">{r["PF"]}</td>'
                     f'<td>{r["累计收益%"]}</td><td class="neg">{r["最差单笔%"]}</td></tr>')
        H.append('</table>')
        H.append('<div class="note">止盈从 +1% 放宽到 +5%，PF 从 1.13 提到 1.35、'
                 '平均单笔从 +0.15% 提到 +0.71% —— <b>+1% 出场过紧，把大部分利润让出去了</b>，'
                 '这是当前规则里最容易改进的一点。</div>')
        H.append('<h3>止损测试（止盈 +1%，持有 5 日）</h3>')
        H.append('<table><tr><th>止损</th><th>胜率%</th><th>平均收益%</th><th>PF</th>'
                 '<th>累计收益%</th><th>最差单笔%</th></tr>')
        for _, r in Gs.sort_values('止损%').iterrows():
            H.append(f'<tr><td>{"不止损" if r["止损%"] == 0 else f"-{int(r["止损%"])}%"}</td>'
                     f'<td>{r["胜率%"]}</td><td>{r["平均收益%"]}</td><td>{r["PF"]}</td>'
                     f'<td>{r["累计收益%"]}</td><td class="neg">{r["最差单笔%"]}</td></tr>')
        H.append('</table>')
        H.append('<div class="note bad"><b>加止损反而更差。</b>'
                 '-3% 止损把 PF 从 1.13 打到 0.87，-5% 止损 0.99 —— '
                 '这类「缩量回调后放量」的票本来就常常先挖坑再反弹，'
                 '止损砍掉的多是后来能涨回来的单子。真正要控的是仓位，不是止损位。</div>')

    H.append('<h2>说明与局限</h2><ul>'
             '<li>股票池按 2026-08 日均成交额取前 300，含轻微前视；实盘用当日快照，会有池子漂移。</li>'
             '<li>买入价取信号 bar 的收盘价，未计滑点与冲击成本；实际成交价通常更差。</li>'
             '<li>出场按日 K 收盘判定，默认可以理想成交。</li>'
             '<li>手续费按单边万 2.5 + 印花税千 1（卖出）估算，单笔约 -0.15%，'
             '平均收益低于 0.3% 的组合在实盘会显著退化。</li>'
             '<li>15min 数据来自 baostock 前复权，日 K 同为 baostock 前复权，两者口径一致。</li>'
             '</ul>')

    H.append('</div></body></html>')
    f = OUT / 'report.html'
    f.write_text('\n'.join(H), encoding='utf-8')
    print(f'报告已生成: {f}')
    print(f'全样本 Top5: PF={pf:.2f} 平均={avg:+.2f}% 胜率={float(r_top5["胜率%"]):.1f}% '
          f'盈利月份={win_months}/{tot_months}')
    print(f'大盘过滤后: PF={mkt_pf:.2f} 交易数={mkt_n}')


if __name__ == '__main__':
    main()

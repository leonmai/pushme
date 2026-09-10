"""
PushPlus 微信推送 (可选组件)
===========================
把盯盘 / 持仓结果推送到个人微信。

用法:
  from push_notify import push_html, push_text
  push_html("标题", "<h1>...</h1>")     # 富文本 (template=html)
  push_text("标题", "纯文本一行...")      # 纯文本 (template=txt)

Token 来源 (优先级):
  1. 显式传入 push_html(token=...)
  2. 环境变量 PUSHPLUS_TOKEN
  没有 token 时所有函数**静默返回 False**, 不影响主流程 (本地跑也不报错)。

GitHub Actions 部署时: 在仓库 Settings → Secrets 里加 PUSHPLUS_TOKEN,
workflow 通过 secrets 注入为环境变量, 代码无需改动。
"""
import os
import json

try:
    import requests
except Exception:
    requests = None

PUSHPLUS_URL = 'https://www.pushplus.plus/send'


def _token(token=None):
    return token or os.environ.get('PUSHPLUS_TOKEN', '').strip()


def push_html(title: str, html: str, token: str | None = None) -> bool:
    """推送富文本 HTML 到微信。成功返回 True。"""
    tok = _token(token)
    if not tok:
        print('[push] 未配置 PUSHPLUS_TOKEN, 跳过微信推送 (不影响主流程)')
        return False
    if requests is None:
        print('[push] 缺少 requests 库, 跳过推送')
        return False
    try:
        r = requests.post(PUSHPLUS_URL, json={
            'token': tok,
            'title': title,
            'content': html,
            'template': 'html',
            'channel': 'wechat',
        }, timeout=20)
        try:
            d = r.json()
        except Exception:
            print(f'[push] 推送返回非 JSON: HTTP {r.status_code}')
            return False
        if d.get('code') == 200:
            print('[push] 微信推送成功')
            return True
        print(f"[push] 推送失败: code={d.get('code')} msg={d.get('msg')}")
        return False
    except Exception as e:
        print(f'[push] 推送异常: {e}')
        return False


def push_text(title: str, text: str, token: str | None = None) -> bool:
    """推送纯文本到微信。成功返回 True。"""
    tok = _token(token)
    if not tok:
        print('[push] 未配置 PUSHPLUS_TOKEN, 跳过微信推送 (不影响主流程)')
        return False
    if requests is None:
        print('[push] 缺少 requests 库, 跳过推送')
        return False
    try:
        r = requests.post(PUSHPLUS_URL, json={
            'token': tok,
            'title': title,
            'content': text,
            'template': 'txt',
            'channel': 'wechat',
        }, timeout=20)
        try:
            d = r.json()
        except Exception:
            print(f'[push] 推送返回非 JSON: HTTP {r.status_code}')
            return False
        if d.get('code') == 200:
            print('[push] 微信推送成功')
            return True
        print(f"[push] 推送失败: code={d.get('code')} msg={d.get('msg')}")
        return False
    except Exception as e:
        print(f'[push] 推送异常: {e}')
        return False


if __name__ == '__main__':
    # 本文件自测: 无 token 时应当静默跳过
    ok = push_html('自测', '<p>hello</p>')
    print('result =', ok)

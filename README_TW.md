
# VibeVoice 本地 OpenAI 相容音訊 API（繁體中文）

本文件說明如何以本地方式啟動 VibeVoice 服務，並透過「OpenAI 官方 Python/JS 客戶端」呼叫語音合成（TTS），同時支援 SSE 串流、voice YAML 映射、參考音檔上傳等功能。

## 快速開始

1) 安裝（在 repo 根目錄）

```bash
uv pip install -e .
# 或 pip install -e .
```

2) 啟動伺服器（預設不需 API Key）

```bash
python -m vibevoice_api.server --model_path "F:/VibeVoice-Large" --port 8000
```

### API 基底路徑（預設 `/v1`）

- 伺服器預設在 `/v1` 前綴下提供所有路由。若需更改，請在啟動前設定 `VIBEVOICE_API_BASE_PATH`（請包含前導斜線 `/`）：

  ```bash
  export VIBEVOICE_API_BASE_PATH=/api
  python -m vibevoice_api.server --model_path "F:/VibeVoice-Large" --port 8000
  ```

- 客戶端請使用相同前綴建立 URL，例如：

  ```python
  base_path = "/api"  # 與 VIBEVOICE_API_BASE_PATH 相同
  client = OpenAI(base_url=f"http://127.0.0.1:8000{base_path}", api_key="ignored")
  ```

- 靜態網頁主控台位於 `<base_path>/web/console.html`。舊的無前綴路徑仍保留相容性，但建議改用明確的前綴。

3) 測試（兩種方式擇一）

方式 A — 使用 OpenAI 官方套件（pip install openai ≥ 1.40）

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",  # 若調整 VIBEVOICE_API_BASE_PATH 請同步修改
    api_key="ignored",
)

speech = client.audio.speech.create(
    model="F:/VibeVoice-Large",
    voice="Alice",
    input="Hello from VibeVoice!",
    response_format="wav",
    instructions="請以溫暖沉穩的主持人口吻說話",
)

with open("out.wav", "wb") as f:
    f.write(speech.read())

方式 B — 純 HTTP 腳本（不依賴 openai）

```bash
python scripts/api_audio_speech_test.py \
  --base_url http://127.0.0.1:8000 \
  --model_path "F:/VibeVoice-Large" \
  --voice alloy \
  --response_format mp3 \
  --out outputs/api_http/out.mp3
```
```

- Base URL：若設定 `VIBEVOICE_API_BASE_PATH`，請在 openai `base_url`、腳本 `--base_url` 或瀏覽器網址中加入相同前綴（例如 `http://127.0.0.1:8000/api`）。
- `response_format`: `wav`, `pcm` 原生；`mp3`, `opus`, `aac` 需伺服器安裝 ffmpeg（flac 已移除）。
- `instructions`: 系統提示/風格指令，長度可由 `VIBEVOICE_INSTRUCTIONS_MAXLEN` 設定（預設 2000），超過會被截斷並在回應 `X-Hints` 顯示 `instructions_clamped`。

4) JS（Node 18+）SSE 客戶端

```bash
node scripts/js/openai_sse_client.mjs   --base http://127.0.0.1:8000   --model "F:/VibeVoice-Large"   --voice Alice   --text "Hello SSE"   --out outputs/js_sse/out.wav
```

此範例會以 `stream_format="sse"` 呼叫 `/audio/speech`，接收 base64 PCM 分塊並輸出成 WAV。

## 參考音檔與 voice 設定

- voice 名稱（掃描 demo/voices）：預設會掃描 `demo/voices/*.wav`，並可用名稱（例如 `en-Alice_woman`）。
- voice YAML 映射：可用 YAML 管理別名（見下一節）。
- 明確路徑：用 `extra_body={"voice_path": "./my_ref.wav"}`。
- 上傳音檔：用 `extra_body={"voice_data": "data:audio/wav;base64,..."}` 或純 base64 字串。

## Voice YAML 映射

伺服器每次請求會重新載入 YAML，變更立即生效。搜尋順序：

1. `VIBEVOICE_VOICE_MAP`（相對 repo root 或絕對路徑）
2. `<repo>/voice_map.yaml`
3. `<repo>/config/voice_map.yaml`

建立範例檔：

```bash
cp config/voice_map.yaml.sample config/voice_map.yaml
```

編輯 `voice_map.yaml`：

```yaml
# 將常見名稱對應到掃描到的聲音名稱
alloy: en-Frank_man
ash: en-Carter_man

# 或以 aliases 區塊對應到檔案路徑或其他名稱
aliases:
  promo_female: demo/voices/en-Alice_woman.wav
  win_custom: F:\voices\my_voice.wav
```

呼叫時直接使用 `voice="alloy"` 或 `voice="promo_female"` 即可。

## SSE 串流（text/event-stream）

- 在請求中加入 `stream_format="sse"`，伺服器會回應 SSE 事件：
  - `event: start` → `{ "format": "pcm", "sample_rate": 24000 }`
  - `event: chunk` → `{ "type": "audio_chunk", "format": "pcm", "data": "<base64 PCM16>" }`
  - `event: end`
- 目前串流僅支援 PCM；若需 mp3/flac 串流，可再討論整合持續編碼（ffmpeg pipe）。

## 觀測與日誌

- `/metrics`：Prometheus 指標（請求數、延遲、合成延遲、活躍推理數、錯誤數、Hints 計數等）。
- `logs/requests.log`：每請求 JSONL（含 request id、model、format、voice、prompt、instructions 等）。
- `logs/hints.log`：Hints JSONL（例如 `speed_clamp`、`instructions_applied`、`ffmpeg_encode`）。
- 回應標頭：`X-Request-ID`、`X-Hints`、`X-Model`、`X-Sample-Rate`、（若夾取）`X-Speed-Clamped: 1`。

## 常見環境變數

- `VIBEVOICE_DEVICE=cpu|cuda|mps`
- `VIBEVOICE_FFMPEG=/path/to/ffmpeg`
- `VIBEVOICE_VALIDATE_SHARDS=0|1`（預檢本地分片）
- `VIBEVOICE_MAX_CONCURRENCY`（單模型同時推理數）
- `VIBEVOICE_LOG_DIR`、`VIBEVOICE_LOG_PROMPTS=0|1`、`VIBEVOICE_PROMPT_MAXLEN`、`VIBEVOICE_INSTRUCTIONS_MAXLEN`
- `VIBEVOICE_VOICE_MAP`（voice_map.yaml 路徑）

## 疑難排解

- 500 並附帶 `IndexError`：已修復並發問題；若仍有，請降低 `VIBEVOICE_MAX_CONCURRENCY`。
- mp3 等格式 400：請安裝 ffmpeg 或設定 `VIBEVOICE_FFMPEG`。
- 找不到聲音檔：確認 YAML 路徑正確、或檔案存在；Windows 路徑請使用雙反斜線 `\`。

---

如需更進階功能（例如真實 mp3/flac 串流、Triton 自訂算子優化），歡迎提出，我們可以協助規劃與實作。

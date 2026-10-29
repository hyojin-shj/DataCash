[English](README.md) · [한국어](README.ko.md) · [Español](README.es.md) · [中文](README.zh-CN.md) · **日本語**

# DataCash

**みんながすでに持っているスマホとスマートウォッチだけで、ロボットが学ぶ人の手のデータを集める。**

人々は自宅で、ロボットの基本動作ミッション（ティッシュを引き出す・コップを移す・ふたを回すなど全 12 種類）をスマホとスマートウォッチで撮影し、報酬を受け取ります。
ヒューマノイドやフィジカル AI の企業が購入するのは、個人情報を取り除いた映像・音・手首の動き・握る力・心拍を一つの時間軸にそろえ、
そのまま学習に使える **LeRobot v3.0** 形式のデータセットです。

Meta Global AI Developer Hackathon（Indie Developer · Push the Frontier）出品作。

![DataCash](docs/1026_datacash_brand.png)

## ひと目で

| | |
|---|---|
| 撮影する人 | Android アプリ + ウォッチアプリ。ミッションを選ぶ → 顔の確認 → 撮影 → 質問に 1〜2 個答える → 「チャリン」と報酬 |
| 購入する企業 | Web で**ふつうの言葉で依頼**（「倉庫ロボット用の箱を持ち上げるデータ、20〜50 代、予算 3,000 ドル」）→ AI がミッションを設計して世界中に公開 |
| プライバシー | 参加者の顔は**実在しない人**に置き換え、同意していない人・画面・書類は隠し、処理が終わったら元の映像は削除 |
| 触覚 | 手袋なしで —— **ウォッチが自ら振動し、手がその振動をどれだけ吸収するか**で握る力を推定（GripSense、UIST 2012） |

## Meta のモデルをどこに使っているか

| モデル | 役割 | コード |
|---|---|---|
| **SAM 3.1** | 同意していない人・子ども・画面・鏡・写真立て・書類を隠す（動画追跡）、扱っている物体の領域 → `observation.objects` | `datacash/meta/sam.py`、`datacash/privacy/mask.py` |
| **Muse Spark 1.3** | 隠した映像 → 手順・指示文・使う手・物体・ミス（構造化 JSON）、企業のふつうの言葉の依頼 → ミッション設計 | `datacash/understand/`、`datacash/requests.py` |
| **Muse Image** | 実在しない 34 人（7 地域 × 男女 × 20〜60 代）—— 顔の置き換え用 | `scripts/make_donors.py`、`datacash/donors/` |
| **Muse Voice Transcribe** | 録音の中で話された個人情報を見つけて消音 | `datacash/privacy/speech.py` |
| **Muse Glimmer**（オープンウェイト） | 30B モデルをこの Mac で llama.cpp により実行 —— Spark の手順をフレームごとに再確認（API 費用なし） | `datacash/understand/glimmer.py`, `datacash/evolve.py` ([docs/model_evolution.md](docs/model_evolution.md)) |
| **Muse Spark Open**（オープンウェイト、816B） | 異議申し立てを見直す「より高いモデル」、すべてのラベルへのセカンドオピニオン（Fireworks）— 両モデルが一致した手順が Glimmer の学習の正解になる | `datacash/understand/spark_open.py`、`scripts/crosscheck.py` |

API を呼ぶたびに費用を `data/api_ledger.jsonl` に記録し、上限を超えたらその段階だけをスキップします。

## 主な技術

### 1. データの価値を守るプライバシー処理
- **ぼかしではなく顔の置き換え：** 参加者の顔を Muse Image で作った人物に置き換えます（LivePortrait + MediaPipe）。同じ性別・近い年齢の中でいちばん似た人を選び、20 フレームを試しに置き換え、顔認識（SFace）が半分以上を「同じ人」と判断するなら次の人へ。
- **連続性：** 置き換えた顔が似すぎたフレームは、モザイクではなく表情を弱めて作り直します —— 映像が途切れません。
- **隠す：** SAM 3.1 で同意のないものだけ —— ほかの人・子ども・画面・鏡・写真立て・書類。同意した家族は隠さず、一人ひとり別の仮想の顔に。
- **元データの削除：** サービスでは処理が終わるとすぐに元の映像・音を削除します（`purge_originals`）。ハッカソンのデモ用サーバーでは、元の映像と置き換え後の比較を見せるため、この呼び出しだけを止めています。

### 2. センサーの自動同期
手をたたく必要はありません：スマホ↔ウォッチの時計の往復、18〜20 kHz の音響ビーコン、ウォッチの振動パルス。実際のスマホ + ウォッチの録画で残差 0.5〜2.3 ミリ秒（`datacash/sync/`）。

### 3. 手袋なしの触覚 —— 手首の GripSense
Goel・Wobbrock・Patel の *GripSense*（UIST 2012）を手首に移しました。ウォッチが短く振動すると、強く握るほど手が振動を吸収し、ジャイロの高周波が減り、指が踏ん張る 4〜12 Hz の震えが増えます。一人ひとりが自分で区別した強さ（離す・軽く・ふつう・ぎゅっと、はかり不要）で学習した決定木が、握る力のレベル 0〜3 を出します（`datacash/touch/`）。 Galaxy Watch4 で実測（IMU は最大 100 Hz、キャリブレーション 1 回、1 回目で学習して 2 回目を当て、逆も）：決定木 25 本 + 2 秒まとめで 4 段階 59%（当てずっぽう 25%、ぎゅっと 100%）、映像が手と物の接触を見ている間は回帰木 50 本が握る強さを 0〜100 の点数で出し（例：「中 (57)」）、低・中・高に区切ると 88%、点数の平均誤差 15 点。
心拍・拍動間隔・皮膚温度は Samsung Health Sensor SDK から（`docs/samsung_health_sdk.md`）。

### 4. ふつうの言葉の依頼 → ミッション
企業が必要なものを書くと、Muse Spark（構造化出力）がミッション（動作・物体・変える条件）、参加条件（年齢・性別）、センサーの組み合わせを設計し、サーバーが予算に合わせて見積もりを出します。確定すると、条件に合う人のアプリにだけミッションが表示されます —— 依頼そのものが収集の分布になります。

### 5. 利益が残る処理コスト
- 使えない録画（短すぎる・手が見えない・暗すぎる）は、**API を呼ぶ前に**ローカルで差し戻し —— 0 ドル
- ローカルの物体検出で先に確認し、SAM には必要なものだけ（一人で撮った録画では「person」を問い合わせない、画面は映っているときだけ、動かない物は毎秒 1 フレーム）
- 実際の録画 4 本：SAM の費用 **2.92 → 0.27 ドル（−91%）**、映像 1 時間あたり約 5〜7 ドル（販売価格 約 43 ドル/時間）
- SAM マスクのデコードを Numba で書き直し：録画 1 本あたり 593 秒 → 8 秒、公式パーサーとビット単位で一致

## データセット（LeRobot v3.0）

`observation.images.ego`（隠した + 顔を置き換えた映像）· `observation.hands` · `observation.body` · `observation.imu.phone` · `observation.imu.wrist` ·
`observation.audio` · **`observation.touch`**（接触、握る強さ 低・中・高 + 0〜100 点）· `observation.touch.vibration` · **`observation.physio`**（心拍、皮膚温度。生体情報に同意した人のみ）·
`observation.objects` · `annotation.step` · `annotation.primitive`（reach · grasp · … · release）· `sync.error_ms` + 品質証明書。
実際の `lerobot` 0.6.1 で読み込めます（`scripts/validate_lerobot.py`）。

企業はデータ 1 時間あたりのセンサーの組み合わせを選んで購入します —— 映像（必須）₩35,000 · 姿勢 ₩20,000 · 音 ₩5,000 · 手首 IMU ₩15,000 · 触覚 ₩25,000 · 生体情報 ₩50,000（`datacash/pricing.py`）。

## 実行

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" torch pyyaml rich
echo "MODEL_API_KEY=..." > .env                       # リポジトリには入れない
# data/models：MediaPipe hand/pose/face landmarker .task、efficientdet_lite0.tflite、OpenCV yunet.onnx + sface.onnx
.venv/bin/python -m pytest                            # 正解がわかっている合成セッションで検証
DATACASH_TOKEN=... .venv/bin/python -m datacash.server.app --port 8800   # サーバー + 企業向け Web
cd app && ./gradlew assembleDebug                     # Android アプリ + ウォッチアプリ
```

LivePortrait はリポジトリに含めていません —— [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait)（MIT）を入手し、`LP_ROOT` で指定してください。

## フォルダ

| フォルダ | 役割 |
|---|---|
| `app/` | Android アプリ（同意・ミッション・顔の確認・撮影・ミスの記録・質問・アップロード・報酬・異議申し立て）+ ウォッチアプリ |
| `web/` | 企業向け Web（ふつうの言葉の依頼・センサーのタイムライン・見積もり・注文・ダウンロード）と運営画面 |
| `datacash/privacy` | SAM で隠す · 顔の置き換え · 音声のプライバシー · 事前スキャン |
| `datacash/sync` · `touch` · `labels` | センサー同期 · 触覚 · 手と体の姿勢 |
| `datacash/understand` | Muse Spark による理解 · Glimmer による確認 · センサーによる検証 |
| `datacash/export` · `jobs.py` | LeRobot 出力 · 証明書 · 処理の流れ |
| `datacash/server` | 収集サーバー + 企業向け API |

## ドキュメント（韓国語）

- ハッカソン計画：[docs/1026_meta_hackathon.md](docs/1026_meta_hackathon.md) · 初日の記録：[docs/DAY1.md](docs/DAY1.md)
- 隠す処理と顔の置き換えの評価：[docs/privacy_eval.md](docs/privacy_eval.md) · Meta モデルの確認：[docs/model_check.md](docs/model_check.md)
- ウォッチ SDK：[docs/samsung_health_sdk.md](docs/samsung_health_sdk.md) · デモ動画の台本：[docs/demo_video_script.md](docs/demo_video_script.md)

## ハッカソン前 / ハッカソン中

| | 内容 |
|---|---|
| 前（10 月 26 日まで） | 計画書とブランド画像のみ。コードなし |
| 中（韓国時間 10 月 27 日 01:00 から） | このリポジトリのすべてのコード、アプリ、Web、Muse Image の人物、ドキュメント |

## ライセンスと出典

MediaPipe（Apache 2.0）· OpenCV YuNet と SFace（Apache 2.0）· LivePortrait（MIT）· LeRobot（Apache 2.0）· Numba（BSD）。
テスト映像は Wikimedia Commons（CC BY・CC BY-SA・パブリックドメインのみ、出典は `docs/privacy_eval.md`）のもので、リポジトリには入れていません。
Samsung Health Sensor SDK は規約に従いリポジトリに入れていません。

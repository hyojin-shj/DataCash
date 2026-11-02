[English](README.md) · [한국어](README.ko.md) · [Español](README.es.md) · [中文](README.zh-CN.md) · **日本語**

## DataCash Promotion Video

<p align="center">
  <a href="https://youtu.be/gw3gqoHlFvA"><img src="docs/demo_video.jpg" width="80%" alt="DataCash 紹介動画 (2:35)"></a>
</p>

## DataCash Technique Video

<p align="center">
  <a href="https://youtu.be/o1sGUME0GHw"><img src="docs/demo_3min.en.jpg" width="80%" alt="DataCash 3 分の技術デモ"></a>
</p>

<p align="center">3 分の技術デモ · <a href="https://youtu.be/o1sGUME0GHw">English</a> · <a href="https://youtu.be/vClUIqrZjBY">한국어</a></p>

---

# DataCash（データキャッシュ）

**Physical AI データプロジェクト**<br>
**家にすでにあるスマホとスマートウォッチだけで、ロボットが学ぶ人の手のデータを集める。**

人々は自宅で、ロボットの基本動作ミッションをスマホとスマートウォッチで撮影し、報酬を受け取ります。<br>
ヒューマノイドやフィジカル AI の企業は、**5感のデータ（視覚・聴覚・触覚・動き・生体）** を一つの時間軸にそろえた **LeRobot v3.0** 形式のまま購入できます。

Meta Global AI Developer Hackathon (Indie Developer)

<p align="center">
  <img src="docs/1026_datacash_brand.png" width="90%" alt="DataCash">
</p>

---

## 主な技術

### 1. 仮想人物への置き換え

- **顔の置き換え：** ぼかしの代わりに、参加者の顔を Muse Image で作った仮想人物に置き換えます（LivePortrait + MediaPipe）。<br>
  同じ性別・近い年齢から候補を選び、20 フレームを試しに置き換え、顔認識（SFace）が「同じ人」と判断するフレームが半分を超えたら次の人で置き換えます。
- **連続性：** 置き換えた顔が似すぎたフレームは、モザイクではなく表情を弱めて作り直します。
- **隠す：** SAM 3.1 で、データ収集に同意していない人・子ども・画面・鏡・写真立て・書類を隠します。
- **元データの削除：** サービスでは処理が終わるとすぐに元の映像・音を削除します。<br>
  ハッカソンのデモ用サーバーでは、元映像↔置き換え後の比較を見せるため、この呼び出しだけを止めています。

#### 顔置き換えの改良の歩み

| 日付 | 変更 | 結果 |
|---|---|---|
| 10-27 | ぼかしの代わりに仮想人物へ置き換え（LivePortrait、Muse Image の仮想人物 34 人）、フレームごとに再識別チェック | 顔は残り、人だけが変わる · 387 フレーム中モザイク 3 |
| 10-27 | 速度：Mac GPU で遅い Conv3d を 2D 7 回に分割 + fp16 | 録画 1 本 636 → 84 秒 |
| 11-01 | 横顔：同じ仮想人物の角度写真 24 枚（Muse Image 編集）→ 場面の角度に合う写真を元に | うつむき・横顔の歪みが減少 |
| 11-01 | 仮想人物の選び方：同じ性別・年代・グループの中で、候補ごとに試行フレームを実際に置き換え、参加者に最も似ていない人を選ぶ（東アジアの仮想人物 36 人追加、$9） | 厳しい再識別で認識される場面 3.3 / 11.1 / 17.3 / 30.9% → 2.4 / 6.4 / 1.7 / 5.0% |

<p align="center">
  <img src="docs/faceswap_views.jpg" width="85%" alt="仮想人物 4 人の角度写真（正面 + 8 枚）"><br>
  <sub>仮想人物ごとに、同じ人の複数の角度の写真</sub>
</p>

<p align="center">
  <img src="docs/faceswap_angle_source.ja.png" width="85%" alt="正面写真 1 枚から作った横顔・うつむき vs 角度の近い写真から作った同じ場面"><br>
  <sub>正面写真 1 枚から作った横顔 vs 角度の近い写真から作った同じ場面</sub>
</p>

<p align="center">
  <img src="docs/faceswap_donor_pick.ja.png" width="85%" alt="前：写真が最も似ている人 / 後：同じ性別・年代・グループで試して最も似ていない人、認識される場面の割合"><br>
  <sub>仮想人物の選び方の前・後：認識される場面の割合</sub>
</p>

<p align="center">
  <img src="docs/faceswap_more_donors.ja.png" width="85%" alt="新しい仮想人物と認識される場面の前・後"><br>
  <sub>仮想人物を増やした後の、認識される場面の前・後</sub>
</p>

### 2. 手袋なしの触覚 —— GripSense 方式

Goel・Wobbrock・Patel の *GripSense*（UIST 2012）を手首に移し、触覚の測定に使いました。<br>
ウォッチが短く振動すると、強く握るほど手が振動を吸収してジャイロの高周波が減り、指が踏ん張る 4〜12Hz の震えが大きくなります。

<p align="center">
  <img width="560" alt="ウォッチの振動で握る力を読み取る仕組み" src="https://github.com/user-attachments/assets/cf02f9f1-a91b-472a-8073-cc072cd7790a" />
</p>

**Galaxy Watch4 での実測**（IMU 最大 100Hz、個人キャリブレーション録画 3 本、1 回目で学習 → 2 回目で評価、逆方向でも交差検証）

- **分類：** ランダムフォレスト（木 25 本）+ 2 秒の平滑化で 4 段階（離す・軽く・ふつう・ぎゅっと）の精度 **59%**（当てずっぽう 25%、「ぎゅっと」93%）
- **回帰：** 映像が手と物体の接触を検出した区間では、回帰フォレスト（木 50 本）が握る力を 0〜100 の連続スコアで推定（例：「中 (57)」）<br>
  → 低・中・高の 3 区間に分けると **83%**、平均絶対誤差（MAE）15 点
- ウォッチの心拍・拍動間隔・皮膚温度は Samsung Health Sensor SDK で取得します（`docs/samsung_health_sdk.md`）。

<p align="center">
  <img src="docs/grip_quality.ja.png" width="85%" alt="最初に一度だけ強さ合わせ → 自分専用モデル → すべての録画で握る力 0〜100；学習していない回で 軽く 97% 低、ふつう 56% 中、ぎゅっと 95% 高で正解">
</p>

### 3. 5感データの同期

スマホ↔ウォッチの時計の往復、18〜20kHz の音響ビーコン（高周波音）、ウォッチの振動パルスで、機器間のデータの同期を合わせます。

- 実際のスマホ + ウォッチの録画 37 本のうち **24 本を音響ビーコンで**合わせました（ペア残差 0.1〜1.9ms）。
- 残りはサーバー経由の時計で合わせますが、この時計はビーコンより 24.7ms（±3.5）ずれているため、そのずれを機器の組み合わせごとに測って差し引きます。
- 結果：ウォッチ ↔ 映像の時間誤差 **24.7 → 3.7ms**（販売基準 10ms）

<p align="center">
  <img src="docs/sync_quality.ja.png" width="85%" alt="ウォッチと映像の時間ずれ：補正前 24.7ms、拍手同期 最大 16.7ms、現在 3.7ms（販売基準 10ms）">
</p>

### 4. Spark 1.3 のラベリング精度向上

区間ごと・動作ごとのラベリング精度を高めるために

1. 映像を 20 秒の断片・0.5 倍速に分け、**5 回ラベリングして比較検証**（結果を合わせて一つの順序を選ぶ）
2. オープンウェイトの **Spark Open 816B で再検証**（動作が切り替わる境界のフレームを選び直す）
3. **人手によるラベリング検証**
4. **Spark を使ったミッションごとのラベルの自動生成**

EPIC-KITCHENS-100 の未使用 8 区間（人手の正解 271 ステップ）基準で、ラベリング精度 **36% → 54%（+18%p）**、境界誤差 **430 → 182ms**。

> Muse Wonderful を使った再検証も検討しましたが、韓国で提供されていないため、オープンウェイトモデルを使いました。

### 5. Glimmer を活用した API コスト削減

ラベリングを毎回 Muse Spark API に任せずに済むよう、オープンウェイトの Muse Glimmer 30B を自前のデータで学習させました（Fireworks LoRA）。<br>
Spark がラベルの下書きを作り、人が確認・修正したステップを正解として Glimmer が学びます。

- 学習前 **41.8%** → 確認済みの正解で学習 **54.1%** → ミッションごとの動作順序を使った解法を追加 **64.1%**
- 今後データが蓄積されるほど Glimmer が判断を引き継ぎ、Spark API の呼び出しを減らして、事業の利益率を高められます。

### 6. SAM による物体判断のコスト削減

- 使えない録画（短すぎる・手が見えない・暗い）は、**API を呼ぶ前に**この Mac で差し戻します。
- この Mac の物体検出で先にざっと確認し、SAM には必要なものだけ判断を任せます。<br>
  （一人で撮った場合は「person」を問い合わせない、画面は映っているときだけ、動かない物は毎秒 1 フレーム）
- 実際の録画 4 本：SAM の費用 **$2.92 → $0.27（−91%）**、映像 1 時間あたり約 $5〜7（販売価格 ₩60,000/時間（約 $43））
- SAM マスクのデコードを Numba に置き換え、処理時間 **593 秒 → 8 秒**

---

## LeRobot 販売データ (LeRobot v3.0)

<p align="center">
  <img width="720" alt="購入企業向け Web の LeRobot v3.0 ビュー" src="https://github.com/user-attachments/assets/8131397c-a152-4891-8100-d7066f90d2fa" />
</p>

| 項目 | 内容 |
|---|---|
| 映像 | `observation.images.ego`（隠した・置き換えた映像） |
| 関節 | `observation.hands` · `observation.body` · `observation.face` |
| 動き | `observation.imu.phone` · `observation.imu.wrist` |
| 音 | `observation.audio` |
| 触覚 | **`observation.touch`**（接触・握る力 低・中・高 + 0〜100 点）· `observation.touch.vibration` |
| 生体 | **`observation.physio`**（心拍・皮膚温度、生体情報に同意した場合のみ） |
| 物体・ラベル | `observation.objects` · `annotation.step` · `annotation.primitive`（reach·grasp·…·release） |
| 同期 | `sync.error_ms` + 品質証明書 |

実際の `lerobot` 0.6.1 で読み込めます（`scripts/validate_lerobot.py`）。

センサーは組み合わせを選んで購入します（1 時間あたり、`datacash/pricing.py`）。

| 映像（必須） | 関節 | 音 | 手首 IMU | 触覚 | 生体 |
|---|---|---|---|---|---|
| ₩35,000 | ₩20,000 | ₩5,000 | ₩15,000 | ₩25,000 | ₩50,000 |

---

## 構成

**全体の流れ** — 企業が言葉で依頼するとミッションが開き、参加者がスマホとウォッチで撮ってアップロードすると、サーバーが処理してデータを納品し、報酬を支払います。

<p align="center"><img src="docs/architecture.en.svg" width="95%" alt="全体構成"></p>

**録画 1 本の処理順**（`datacash/jobs.py`）— 費用のかからない段階を先に行い、使えない録画は Meta モデルを呼ぶ前に差し戻します。

<p align="center"><img src="docs/pipeline.en.svg" width="95%" alt="処理順"></p>

**技術スタックとデプロイ** — 現在は Docker やクラウドを使わず、Mac 1 台がサーバーと処理をすべて担い、アプリとウォッチは同じ Wi-Fi 上で mDNS によりサーバーを見つけて直接アップロードします。

<p align="center"><img src="docs/tech_stack.en.svg" width="95%" alt="技術スタック"></p>

## 実行

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" torch pyyaml rich
echo "MODEL_API_KEY=..." > .env                       # リポジトリには入れない
# data/models に MediaPipe hand/pose/face landmarker .task、efficientdet_lite0.tflite、OpenCV yunet.onnx·sface.onnx
.venv/bin/python -m pytest                            # 正解がわかっている合成 5感セッションで検証
DATACASH_TOKEN=... .venv/bin/python -m datacash.server.app --port 8800   # サーバー + 購入企業向け Web
cd app && ./gradlew assembleDebug                     # Android アプリ + ウォッチアプリ
```

## フォルダ

| フォルダ | 役割 |
|---|---|
| `app/` | Android アプリ（同意・ミッション・顔の確認・撮影・ミスの記録・質問・アップロード・報酬・異議申し立て）+ ウォッチアプリ |
| `web/` | 購入企業向け Web（言葉での依頼・5感タイムライン・見積もり・注文・ダウンロード）、運営者画面 |
| `datacash/privacy` | SAM で隠す · 顔の置き換え · 音声のプライバシー · 事前スキャン |
| `datacash/sync` · `touch` · `labels` | 5感同期 · 触覚 · 手と体の関節 |
| `datacash/understand` | Muse Spark による理解 · Spark Open による二次検証 · センサーによる検証 |
| `datacash/export` · `jobs.py` | LeRobot 出力 · 証明書 · 処理順 |
| `datacash/server` | 収集サーバー + 購入企業向け API |

## Meta モデルの利用

| モデル | 用途 |
|---|---|
| **SAM 3.1** | 同意していない人・子ども・画面・鏡・写真立て・書類を隠す、扱っている物体・動作の認識 |
| **Muse Spark 1.3** | 隠した映像 → ステップ・指示文・使う手・物体・ミスのラベリング<br>企業の言葉での依頼 → ミッション設計 → ミッションごとのラベル生成 |
| **Muse Image** | 実在しない仮想人物 70 人を生成（顔の置き換え用）、各人の角度写真 24 枚 |
| **Muse Voice Transcribe** | 録画の音声で個人情報が聞こえた区間を消音 |
| **Muse Spark Open** | 録画の二次検証と異議申し立てを見直す「より高いモデル」。<br>2 つのモデルの見方が違うステップは修正し、一致してセンサーでも検証されたステップは正解例として、次の録画の Spark 1.3 への質問に付きます。 |
| **Muse Glimmer 30B** | Spark 1.3 の下書きと人が検証したラベルで映像認識を学習します。<br>データが集まるほど性能が向上し、API への依存度を下げられます。 |

## 測定した性能

| 項目 | 以前 → 現在 | 測定対象 |
|---|---|---|
| ステップラベルのフレーム精度（Muse Spark 1.3） | **35.6% → 54.3%** | EPIC-KITCHENS-100 未使用 8 区間、人手の正解 271 ステップ |
| ステップ境界誤差（中央値） | **430 → 182ms** | 同上 |
| Muse Glimmer 30B のステップ精度（LoRA、Fireworks） | **41.8% → 54.1% → 64.1%**<br>（v80 人手の正解で学習 · v82 + ミッションごとの順序デコード、範囲 60.7–67.2% · Spark ラベルで学習した v79 は 32.8%） | 学習に使っていない録画 8 本の人手の正解 122 ステップ |
| ウォッチ ↔ 映像の時間誤差 | **24.7 → 3.7ms**（基準 10ms） | スマホ + ウォッチの録画 37 本 |
| 録画 1 本の顔置き換え時間 | **636 → 84 秒**（7.5 倍） | `docs/privacy_eval.md` |
| SAM 3.1 のコスト、録画 4 本 | **$2.92 → $0.27**（−91%） | 実際の録画 4 本 |
| 握る力 低・中・高 | **83%**（偶然 33%） | キャリブレーション録画 3 本、Galaxy Watch4 |

<p align="center">
  <img src="docs/perf/label_accuracy.en.png" width="85%" alt="ステップラベルの精度"><br><br>
  <img src="docs/perf/confusion.en.png" width="85%" alt="混同行列"><br><br>
  <img src="docs/perf/glimmer.en.png" width="85%" alt="Glimmer の学習"><br><br>
  <img src="docs/perf/monitoring.en.png" width="85%" alt="サービス監視">
</p>

**サービス監視**（運営者画面と同じ値、`datacash/perf.py`）— 基準を超えると警告し、その録画はレビュー対象になります。

- 録画ごとの Spark 1.3 ↔ Spark Open の一致率と直近 5 本の平均 — 50% を下回ると警告
- 録画 1 本の処理時間 — 中央値の 2 倍を超えると警告
- 顔の置き換え後も同じ人に見えるフレーム（SFace 0.363）— 5% を超えると警告

---

## ハッカソン前 / ハッカソン中

| | 内容 |
|---|---|
| ハッカソン前（10-26 以前） | 計画書とブランド画像のみ。コードなし |
| ハッカソン中（韓国時間 10-27 01:00 から） | このリポジトリのすべてのコード · アプリ · Web · Muse Image の仮想人物 · ドキュメント |

## ライセンス・出典

MediaPipe (Apache 2.0) · OpenCV YuNet·SFace (Apache 2.0) · LivePortrait (MIT) · LeRobot (Apache 2.0) · Numba (BSD)<br>
テスト映像は Wikimedia Commons（CC BY・CC BY-SA・パブリックドメインのみ、出典は `docs/privacy_eval.md`）<br>
Samsung Health Sensor SDK

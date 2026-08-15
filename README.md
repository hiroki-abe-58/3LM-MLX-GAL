# 3LM-MLX-GAL &mdash; 一晩学習させたモデルに、48秒で口調を移植する

[3LM-MLX](https://github.com/hiroki-abe-58/3LM-MLX)（M1 Max で一晩・8時間
事前学習した 35.66M の日本語モデル）に、**自作の会話 2,610件**で口調を
学習させた記録です。

測りたかったのは、口調が変わることではありません。

> **モデルが大きくなると、必要なキャラクターデータも増えるのか。**

前作（13.81M）は 2,610会話・40秒で口調が変わりました。
今回のモデルはパラメータが 2.6倍、事前学習データが 126倍です。
**同じ 2,610件では足りないだろう**と予想して、足りなければ増産して
境界を測るつもりでした。

**外れました。** 増産の出番はありませんでした。

## 同じ会話データで、前作と今作を並べる

質問も乱数もサンプリング条件もそろえて、走らせた6問を選ばずに全部並べます。

![前作と今作の会話比較](https://raw.githubusercontent.com/hiroki-abe-58/3LM-MLX/main/docs/images/3lm-gal-chat.png)

前作は「猫は元気になることない」「うちはちょっとずつえんでたわ」と
日本語自体が崩れます。今作は「猫、かわいいよね〜」と受けてから逸れます。
**話が逸れる先が同じ**なのは、どちらも同じ 2,610会話を読んでいるからです。

その3の時点ではこう見えていました（端末の画面）。

![その3のギャル会話](https://raw.githubusercontent.com/hiroki-abe-58/2LM-MLX-GAL/main/docs/images/2lm-chat-gal-2.png)

面白いのは、**大きくしても「お腹が空く」癖は消えなかった**ことです。
キャラクターの中身は事前学習ではなく**会話データ側が決めている**、
という当たり前のことが、並べると見えます。

| 置き場 | 中身 |
|---|---|
| [GeneLab/3LM-MLX-GAL](https://huggingface.co/GeneLab/3LM-MLX-GAL) | 口調を学習させた重み |
| [GeneLab/3LM-MLX](https://huggingface.co/GeneLab/3LM-MLX) | 学習前（対話調整のみ） |
| [3LM-MLX](https://github.com/hiroki-abe-58/3LM-MLX) | **学習・評価のコード一式** |
| [2LM-MLX-GAL](https://github.com/hiroki-abe-58/2LM-MLX-GAL) | 会話データを作ったスクリプト（その3） |

**このリポジトリにコードは置いていません。** 学習と評価は 3LM-MLX、
データ生成はその3のものをそのまま使っています。複製すると片方だけ直したときに
気づけなくなるためです。ここにあるのは**会話データと実測結果**です。

## すぐ試す

```bash
pip install mlx numpy sentencepiece huggingface_hub

git clone https://github.com/hiroki-abe-58/3LM-MLX.git
cd 3LM-MLX

python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('GeneLab/3LM-MLX-GAL', local_dir='checkpoints/gal-final')"

python src/chat_cli.py --ckpt checkpoints/gal-final
```

```
こんにちは            -> ねえ、ちょっと寂しいけど、腹減った〜。今日はもう寝たい…
疲れました            -> お疲れさま、リフレッシュしてみよ！うちも元気でいてね
何から始めればいいですか -> まずは簡単なサイトから始めてみて〜。そこから始めてみるのもアリかも！
```

## 口調を乗せる

```bash
# 3LM-MLX の中で実行する。会話データはここから取る
curl -sL -o data/raw/gal_chat.jsonl \
  https://raw.githubusercontent.com/hiroki-abe-58/3LM-MLX-GAL/main/data/gal_chat.jsonl

python data/prepare_sft.py --no-hf --out data/corpus_gal.txt --min-char-freq 1

python src/sft.py --init-from checkpoints/sft-final --corpus data/corpus_gal.txt \
    --out checkpoints/gal --epochs 8 --lr 1e-4
```

**30ステップ、48秒**で終わります。学習率は事前学習の 6e-4 から 1e-4 に落とします。
作った表現を壊さない範囲で、口調だけ動かしたいからです。

気をつけることが3つあります。

**語彙を作り直さない。** `--init-from` を付けるとトークナイザは事前学習した
ものが使われます。作り直すと ID の対応が変わり、事前学習した重みが全部
無意味になります。

**低頻度文字フィルタを切る（`--min-char-freq 1`）。** 既定の 10 は
58,014会話のコーパス用の閾値です。同じ閾値を 2,610件に当てると
母数が22分の1なので条件が急に厳しくなり、**47%（1,234件）が黙って消えます。**
「〜」「♪」や少し珍しい漢字を含む会話ごと落ちるためです。
**閾値は「回数」で書かれていることが多く、コーパスの大きさを変えると意味が変わります。**

**周回数は前作のまま使わない。** 検証損失は step 20（5.3周）が最良で、
step 30（7.9周）では 3.9201 → 3.9965 と悪化しました。
2,610会話を8周は、このモデルには多すぎます。書き出しは最良の重みを選ぶので
実害はありませんでしたが、**前作と同じ回数が最適とは限りません。**

## 結果: 大きいほうが、失うものが少なかった

同じ 2,610会話・同じ検証セット・同じサンプリング条件
（temperature 0.8 / top_k 40 / repetition_penalty 1.15 / seed 777）で、
13.81M と 35.66M の前後を並べます。

| 指標 | 2LM 素 | 2LM ギャル | 3LM 素 | 3LM ギャル |
|---|---|---|---|---|
| bits/char（低いほど良い）| 2.540 | 3.510 | 2.801 | **3.154** |
| 主題保持率（高いほど良い）| 0.733 | 0.267 | 0.667 | **0.400** |
| 反復率 | 0.100 | 0.000 | 0.300 | 0.000 |
| 破綻率 | 0.000 | 0.000 | 0.000 | 0.000 |
| 平均返答長 | 64.1文字 | 25.1文字 | 76.0文字 | 26.1文字 |

口調を得るために払った代償を、差で見ます。

![口調を乗せる代償](https://raw.githubusercontent.com/hiroki-abe-58/3LM-MLX/main/docs/images/3lm-gal-cost.png)

| 代償 | 2LM 13.81M | 3LM 35.66M |
|---|---|---|
| bits/char の悪化 | +0.970 | **+0.353** |
| 主題保持率の低下 | -0.466 | **-0.267** |

**どちらも半分以下**になりました。ギャル化後の絶対値でも 3.154 対 3.510 で
今回が上回ります。この検証セットは前作が事前学習に使った公開データ由来で、
**前作のホームグラウンド**にあたるのですが、そこですら勝っています。

### なぜ逆になったのか

素の状態では「文体を変える」ことが、モデルにとって
**持っている知識を捨てること**とほぼ同義だったのだと思います。
949万文字ぶんの狭い表現しか持っていなければ、その表現を上書きすれば中身も消えます。

3.68億トークンを読んだモデルには余裕があります。口調は表層の付け替えで済み、
下にある日本語の能力は残る。**容量ではなく、事前学習の厚みが効いています。**

### 境界は測れていません

当初の狙いだった「必要な会話数の境界」は**測れていません**。
2,610件で足りてしまったので、下限がどこかは分かりません
（855件なら足りないのか、は今回の実験では答えられません）。
外れた仮説を、答えたふりで埋めないでおきます。

## 会話データについて

`data/gal_chat.jsonl` に 2,610会話（111,574文字 / 文字種 1,407）が入っています。
実在の人物とは関係のない**架空の会話**です。

なお、9会話の本文に特定のメッセージアプリの名前が出てきます。
生成時のプロンプトがその名前を使っていたためで、現在は
プロンプトからもふるいの許可リストからも外してあります。
本文をあとから書き換えると、**公開している重みを学習したデータと
中身が食い違う**ので、9件はそのままにしてあります。

生成には Apache-2.0 のローカルモデル（calm3-22b-chat）だけを使っています。
このモデルは本プロジェクトが学習に使っている `llm-jp/magpie-sft-v1.0` を
作ったモデル本人でもあります。手元で完結するので、出力の扱いに制約が付きません。

**このデータを作ったのはその3**（2LM-MLX-GAL）で、今回は1文字も足していません。
生成と検査のスクリプトも、作った場所にそのまま置いてあります。

| スクリプト | 役割 |
|---|---|
| [`data/gal/generate.py`](https://github.com/hiroki-abe-58/2LM-MLX-GAL/blob/main/data/gal/generate.py) | 話題の列挙 → 往復生成 → 検査してまとめる |
| [`data/gal/validate.py`](https://github.com/hiroki-abe-58/2LM-MLX-GAL/blob/main/data/gal/validate.py) | 定型化・破綻・重複のふるい |
| [`data/gal/README.md`](https://github.com/hiroki-abe-58/2LM-MLX-GAL/blob/main/data/gal/README.md) | 作り方の詳細 |

日本語の中身を人間が書かないのは意図的です。話題の語彙まで用意すると
**そこが多様性の上限**になるので、ジャンルという骨組みだけ決めて、
肉は全部モデルに付けさせています。

## このリポジトリの中身

**会話データ 1本だけです。**

| 場所 | 中身 |
|---|---|
| `data/gal_chat.jsonl` | 会話 2,610件（111,574文字）|

上の表の数字の出どころは 3LM-MLX 側にあります。
同じファイルを2箇所に置くと、片方だけ直したときに気づけなくなるので、
**このリポジトリには複製を置きません。**

| 実測の生データ | 場所 |
|---|---|
| [`runs/eval_3lm_gal.json`](https://github.com/hiroki-abe-58/3LM-MLX/blob/main/runs/eval_3lm_gal.json) ほか | 4指標の採点結果（2LM 素/ギャル、3LM 素/ギャル）|
| [`runs/3lm/gal_loss.csv`](https://github.com/hiroki-abe-58/3LM-MLX/blob/main/runs/3lm/gal_loss.csv) | 口調学習の損失ログ |
| [`tools/plot_gal_cost.py`](https://github.com/hiroki-abe-58/3LM-MLX/blob/main/tools/plot_gal_cost.py) | 上の図を作るコード |

## シリーズ

| リポジトリ | 内容 |
|---|---|
| [1LM](https://github.com/hiroki-abe-58/1LM) | 1時間で動くものを作る |
| [2LM-MLX](https://github.com/hiroki-abe-58/2LM-MLX) | 会話が成立するレベルへ |
| [2LM-MLX-GAL](https://github.com/hiroki-abe-58/2LM-MLX-GAL) | 学習データを自分で作る |
| [3LM-MLX](https://github.com/hiroki-abe-58/3LM-MLX) | 一晩の学習を落とさず完走させる |
| **3LM-MLX-GAL**（ここ）| **厚く事前学習したモデルに口調を乗せる** |

## ライセンス / クレジット

- 会話データ・ドキュメント: MIT
- 重み: Apache-2.0 相当（[GeneLab/3LM-MLX-GAL](https://huggingface.co/GeneLab/3LM-MLX-GAL)）
- 事前学習コーパスの帰属: FineWeb2（ODC-By 1.0）/ 青空文庫版（CC BY 4.0）

詳しい帰属表示は [NOTICE](NOTICE) と
[3LM-MLX](https://github.com/hiroki-abe-58/3LM-MLX) を参照してください。

# Point Cloud Viewer via Web Browser

Stanford Bunnyの点群データを3D表示するWebアプリケーション

## インストール手順

### 1. uvを使用した環境構築

```bash
# python仮想環境の作成
uv venv

# 仮想環境をアクティベート
source .venv/bin/activate

# 依存関係のインストール
uv sync
```

### 2. アプリケーションの起動

#### Streamlit版 (Pydeck使用)

```bash
# Streamlitアプリケーションを起動（デフォルトポート8501）
streamlit run bunny_viewer_streamlit.py

# 特定のポートを指定して起動
streamlit run bunny_viewer_streamlit.py --server.port 8080
```

#### Gradio版 (Plotly使用)

```bash
# Gradioアプリケーションを起動（デフォルトポート7860）
python bunny_viewer_gradio.py

# 特定のポートで起動
python bunny_viewer_gradio.py --port 8080

# 共有リンク付きで起動
python bunny_viewer_gradio.py --share

# ヘルプを表示
python bunny_viewer_gradio.py --help
```

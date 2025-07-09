# Point Cloud Viewer via Web Browser

Streamlit + Open3D + PyDeckを使用したWeb上の点群ビューアーのコード例

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

```bash
# Streamlitアプリケーションを起動（デフォルトポート8501）
streamlit run bunny_viewer_streamlit.py

# 特定のポートを指定して起動
streamlit run bunny_viewer_streamlit.py --server.port 8080
```

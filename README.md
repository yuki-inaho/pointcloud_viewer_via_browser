# Web Point Cloud Viewer

Streamlit + Open3D + PyDeckを使用したWeb上の点群ビューアーのコード例

## インストール手順

### 1. uvを使用した環境構築

```bash
# 依存関係のインストール
uv sync

# 仮想環境をアクティベート
uv shell
```

### 2. アプリケーションの起動

```bash
# Streamlitアプリケーションを起動（デフォルトポート8501）
streamlit run bunny_viewer_streamlit.py

# 特定のポートを指定して起動
streamlit run bunny_viewer_streamlit.py --server.port 8080
```
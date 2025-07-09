import argparse
import gradio as gr
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import Normalize
from dataclasses import dataclass
from typing import Dict, Any, List
from enum import Enum


# ===== 1. 座標変換モジュール =====
class CoordinateSystem(Enum):
    """座標系の種類"""

    OPEN3D = "open3d"  # X=右, Y=上, Z=後ろ (-Z前方)
    PLOTLY = "plotly"  # X=右, Y=奥行き, Z=上 (-Y画面内)


class CoordinateTransformer:
    """座標系変換を管理するクラス"""

    def __init__(self):
        """変換行列を初期化"""
        self._setup_transformation_matrices()

    def _setup_transformation_matrices(self):
        """変換行列を設定

        Stanford Bunnyの正しい向き: 耳が上向きで座っている状態
        Open3D: X=右, Y=上, Z=後ろ
        Plotly: X=右, Y=奥行き, Z=上

        適切な変換: (x, y, z) -> (x, z, y)
        """
        # Open3D -> Plotly変換行列
        # (x, y, z) -> (x, z, y)
        self.open3d_to_plotly = np.array(
            [
                [1, 0, 0],  # X軸はそのまま
                [0, 0, 1],  # Open3DのZ軸 -> PlotlyのY軸
                [0, 1, 0],  # Open3DのY軸 -> PlotlyのZ軸
            ]
        )

        # Plotly -> Open3D変換行列（逆変換）
        # (x, y, z) -> (x, z, y)
        self.plotly_to_open3d = np.array(
            [
                [1, 0, 0],  # X軸はそのまま
                [0, 0, 1],  # PlotlyのZ軸 -> Open3DのY軸
                [0, 1, 0],  # PlotlyのY軸 -> Open3DのZ軸
            ]
        )

    def transform_points(
        self,
        points: np.ndarray,
        from_system: CoordinateSystem,
        to_system: CoordinateSystem,
    ) -> np.ndarray:
        """点群の座標変換

        Args:
            points: 変換する点群 (N, 3)
            from_system: 変換元の座標系
            to_system: 変換先の座標系

        Returns:
            変換後の点群 (N, 3)
        """
        if from_system == to_system:
            return points.copy()

        if (
            from_system == CoordinateSystem.OPEN3D
            and to_system == CoordinateSystem.PLOTLY
        ):
            return self._apply_transformation(points, self.open3d_to_plotly)
        elif (
            from_system == CoordinateSystem.PLOTLY
            and to_system == CoordinateSystem.OPEN3D
        ):
            return self._apply_transformation(points, self.plotly_to_open3d)
        else:
            raise ValueError(
                f"Unsupported transformation: {from_system} -> {to_system}"
            )

    def _apply_transformation(
        self, points: np.ndarray, transform_matrix: np.ndarray
    ) -> np.ndarray:
        """変換行列を適用"""
        return np.dot(points, transform_matrix.T)

    def get_coordinate_system_info(self, system: CoordinateSystem) -> Dict[str, str]:
        """座標系の情報を取得

        Returns:
            座標系の説明辞書
        """
        if system == CoordinateSystem.OPEN3D:
            return {
                "name": "Open3D",
                "x_axis": "右",
                "y_axis": "上",
                "z_axis": "後ろ (-Z前方)",
            }
        elif system == CoordinateSystem.PLOTLY:
            return {
                "name": "Plotly",
                "x_axis": "右",
                "y_axis": "奥行き",
                "z_axis": "上",
            }
        else:
            raise ValueError(f"Unknown coordinate system: {system}")


# グローバルなトランスフォーマーインスタンス
transformer = CoordinateTransformer()


# ===== 2. 設定管理 (Settings) =====
@dataclass
class Settings:
    NUM_POINTS: int = 10000
    POINT_SIZE: int = 1
    COLORMAP: str = "viridis"
    BG_COLOR: str = "#555555"
    GRID_COLOR: str = "rgba(150, 150, 150, 0.4)"
    GRID_LINES: int = 21
    AXIS_LENGTH_RATIO: float = 0.15


SETTINGS = Settings()


# ===== 3. データ構造 (Data Class) =====
@dataclass
class PointCloudData:
    points: np.ndarray
    colors: np.ndarray
    bbox: Dict[str, Any]


# ===== 4. データ処理 (Processor) =====
class PointCloudProcessor:
    """点群データの読み込み、変換、色付けなどデータ処理全般を担当"""

    def __init__(self):
        pass

    def _calculate_bbox(self, points: np.ndarray) -> Dict[str, Any]:
        min_b, max_b = points.min(axis=0), points.max(axis=0)
        size = max_b - min_b
        return {"min": min_b, "max": max_b, "diagonal": np.linalg.norm(size)}

    def _apply_colors(self, points: np.ndarray) -> np.ndarray:
        # Z座標（高さ）に基づいて色付け
        z_coords = points[:, 2]
        norm = Normalize(vmin=z_coords.min(), vmax=z_coords.max())
        return norm(z_coords)

    def load_and_process(self) -> PointCloudData:
        """データを読み込み、処理フロー全体を実行する"""
        mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
        pcd = mesh.sample_points_uniformly(number_of_points=SETTINGS.NUM_POINTS)
        pcd.translate(-pcd.get_center())

        points_o3d = np.asarray(pcd.points)
        # 座標系変換（Open3D → Plotly）
        points = transformer.transform_points(
            points_o3d, CoordinateSystem.OPEN3D, CoordinateSystem.PLOTLY
        )
        bbox = self._calculate_bbox(points)
        colors = self._apply_colors(points)

        return PointCloudData(points=points, colors=colors, bbox=bbox)


# ===== 5. 可視化 (Visualizer) =====
class PlotlyVisualizer:
    """Plotlyのトレースと図の生成を担当"""

    def _create_point_cloud_trace(self, data: PointCloudData) -> go.Scatter3d:
        return go.Scatter3d(
            x=data.points[:, 0],
            y=data.points[:, 1],
            z=data.points[:, 2],
            mode="markers",
            marker=dict(
                size=SETTINGS.POINT_SIZE,
                color=data.colors,
                colorscale=SETTINGS.COLORMAP,
                showscale=True,
                colorbar=dict(
                    title=dict(text="高さ", font=dict(color="white")),
                    tickmode="linear",
                    tick0=0,
                    dtick=0.2,
                    tickfont=dict(color="white"),
                ),
            ),
            hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
        )

    def _create_axis_traces(self, data: PointCloudData) -> List[go.Scatter3d]:
        scale = data.bbox["diagonal"] * SETTINGS.AXIS_LENGTH_RATIO
        axes = []

        # X軸（赤）
        axes.append(
            go.Scatter3d(
                x=[0, scale],
                y=[0, 0],
                z=[0, 0],
                mode="lines",
                line=dict(color="red", width=5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Y軸（緑）
        axes.append(
            go.Scatter3d(
                x=[0, 0],
                y=[0, scale],
                z=[0, 0],
                mode="lines",
                line=dict(color="green", width=5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Z軸（青）
        axes.append(
            go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[0, scale],
                mode="lines",
                line=dict(color="blue", width=5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        return axes

    def _create_axis_labels(self, data: PointCloudData) -> List[go.Scatter3d]:
        """軸ラベル（X, Y, Z）を作成"""
        scale = (
            data.bbox["diagonal"] * SETTINGS.AXIS_LENGTH_RATIO * 1.3
        )  # 軸より長い位置
        labels = []

        # X軸ラベル
        labels.append(
            go.Scatter3d(
                x=[scale],
                y=[0],
                z=[0],
                mode="text",
                text=["X"],
                textfont=dict(color="red", size=18, family="Arial Black"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Y軸ラベル
        labels.append(
            go.Scatter3d(
                x=[0],
                y=[scale],
                z=[0],
                mode="text",
                text=["Y"],
                textfont=dict(color="green", size=18, family="Arial Black"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Z軸ラベル
        labels.append(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[scale],
                mode="text",
                text=["Z"],
                textfont=dict(color="blue", size=18, family="Arial Black"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        return labels

    def _create_grid_trace(self, data: PointCloudData) -> List[go.Scatter3d]:
        size = data.bbox["diagonal"] * 0.7
        half = size / 2.0
        step = size / (SETTINGS.GRID_LINES - 1)
        z_level = data.bbox["min"][2]

        traces = []
        for i in range(SETTINGS.GRID_LINES):
            pos = -half + i * step
            # X方向の線
            traces.append(
                go.Scatter3d(
                    x=[pos, pos],
                    y=[-half, half],
                    z=[z_level, z_level],
                    mode="lines",
                    line=dict(color=SETTINGS.GRID_COLOR, width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Y方向の線
            traces.append(
                go.Scatter3d(
                    x=[-half, half],
                    y=[pos, pos],
                    z=[z_level, z_level],
                    mode="lines",
                    line=dict(color=SETTINGS.GRID_COLOR, width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        return traces

    def create_figure(self, data: PointCloudData) -> go.Figure:
        """全てのトレースを統合し、Figureオブジェクトを生成"""
        traces = [
            self._create_point_cloud_trace(data),
            *self._create_axis_traces(data),
            *self._create_axis_labels(data),
            *self._create_grid_trace(data),
        ]

        fig = go.Figure(data=traces)

        # レイアウトの設定
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=""),
                yaxis=dict(showbackground=False, showticklabels=False, title=""),
                zaxis=dict(showbackground=False, showticklabels=False, title=""),
                bgcolor=SETTINGS.BG_COLOR,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), center=dict(x=0, y=0, z=0)),
            ),
            paper_bgcolor=SETTINGS.BG_COLOR,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            height=600,
        )

        return fig


# ===== 6. Gradioアプリケーション (App) =====
class PointCloudViewerApp:
    """Gradio UIを構築し、全てのコンポーネントを統合してアプリを起動"""

    def __init__(self):
        self.processor = PointCloudProcessor()
        self.visualizer = PlotlyVisualizer()
        # 初期データを読み込み、コンポーネントを事前生成
        self._load_data_and_components()

    def _load_data_and_components(self):
        """データの読み込みと、それに基づくUIコンポーネントの生成"""
        self.data = self.processor.load_and_process()
        self.fig = self.visualizer.create_figure(self.data)

    def _handle_refresh(self):
        """リフレッシュボタンが押されたときの処理"""
        self._load_data_and_components()
        return self.fig

    def launch(self, server_port=7860, share=False):
        """Gradio Blocks UIを構築して起動"""
        with gr.Blocks(title="3D点群ビューア") as demo:
            gr.Markdown("# 3D点群ビューア (Gradio + Plotly版)")
            with gr.Row():
                plot_output = gr.Plot(value=self.fig, elem_id="plotly-container")

            with gr.Row():
                refresh_btn = gr.Button("再生成", variant="primary")

            # 操作方法の表示
            gr.Markdown(
                """
                **操作方法**
                - **回転:** マウスの左ボタンを押しながらドラッグ
                - **移動:** マウスの右ボタンを押しながらドラッグ
                - **ズーム:** マウスホイールをスクロール
                """
            )

            refresh_btn.click(
                fn=self._handle_refresh, outputs=[plot_output], show_progress="full"
            )
        demo.launch(server_port=server_port, share=share)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="高性能3D点群ビューア (Gradio版)")
    parser.add_argument(
        "--port", type=int, default=7860, help="サーバーポート番号 (デフォルト: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true", help="Gradioの共有リンクを有効にする"
    )

    args = parser.parse_args()

    app = PointCloudViewerApp()
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

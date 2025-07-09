import streamlit as st
import open3d as o3d
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit.components.v1 as components
from typing import Tuple, Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from enum import Enum


# ===== 座標変換モジュール =====
class CoordinateSystem(Enum):
    """座標系の種類"""

    OPEN3D = "open3d"  # X=右, Y=上, Z=後ろ (-Z前方)
    PYDECK = "pydeck"  # X=右, Y=上, Z=前方 (-Z画面内)


class CoordinateTransformer:
    """座標系変換を管理するクラス"""

    def __init__(self):
        """変換行列を初期化"""
        self._setup_transformation_matrices()

    def _setup_transformation_matrices(self):
        """変換行列を設定

        Stanford Bunnyの正しい向き: 耳が上向きで座っている状態
        Open3D: X=右, Y=上, Z=後ろ
        Pydeck: X=右, Y=上, Z=前方

        適切な変換: (x, y, z) -> (x, -z, y)
        """
        # Open3D -> Pydeck変換行列
        # (x, y, z) -> (x, -z, y)
        self.open3d_to_pydeck = np.array(
            [
                [1, 0, 0],  # X軸はそのまま
                [0, 0, -1],  # Open3DのZ軸 -> PydeckのY軸（反転）
                [0, 1, 0],  # Open3DのY軸 -> PydeckのZ軸
            ]
        )

        # Pydeck -> Open3D変換行列（逆変換）
        # (x, y, z) -> (x, z, -y)
        self.pydeck_to_open3d = np.array(
            [
                [1, 0, 0],  # X軸はそのまま
                [0, 0, 1],  # PydeckのZ軸 -> Open3DのY軸
                [0, -1, 0],  # PydeckのY軸 -> Open3DのZ軸（反転）
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
            and to_system == CoordinateSystem.PYDECK
        ):
            return self._apply_transformation(points, self.open3d_to_pydeck)
        elif (
            from_system == CoordinateSystem.PYDECK
            and to_system == CoordinateSystem.OPEN3D
        ):
            return self._apply_transformation(points, self.pydeck_to_open3d)
        else:
            raise ValueError(
                f"Unsupported transformation: {from_system} -> {to_system}"
            )

    def _apply_transformation(
        self, points: np.ndarray, transform_matrix: np.ndarray
    ) -> np.ndarray:
        """変換行列を適用"""
        return np.dot(points, transform_matrix.T)

    def transform_axis_vectors(
        self, from_system: CoordinateSystem, to_system: CoordinateSystem
    ) -> Dict[str, np.ndarray]:
        """座標軸ベクトルの変換

        Returns:
            変換後の軸ベクトル辞書 {"x": [1,0,0], "y": [0,1,0], "z": [0,0,1]}
        """
        # 単位ベクトル
        unit_vectors = np.array(
            [
                [1, 0, 0],  # X軸
                [0, 1, 0],  # Y軸
                [0, 0, 1],  # Z軸
            ]
        )

        transformed_vectors = self.transform_points(
            unit_vectors, from_system, to_system
        )

        return {
            "x": transformed_vectors[0],
            "y": transformed_vectors[1],
            "z": transformed_vectors[2],
        }

    def transform_bounding_box(
        self,
        bbox: Dict[str, Any],
        from_system: CoordinateSystem,
        to_system: CoordinateSystem,
    ) -> Dict[str, Any]:
        """境界ボックスの座標変換

        Args:
            bbox: 境界ボックス情報
            from_system: 変換元の座標系
            to_system: 変換先の座標系

        Returns:
            変換後の境界ボックス情報
        """
        if from_system == to_system:
            return bbox.copy()

        # 境界ボックスの8つの頂点を生成
        min_bound = bbox["min"]
        max_bound = bbox["max"]

        corners = np.array(
            [
                [min_bound[0], min_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]],
                [min_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], max_bound[2]],
            ]
        )

        # 頂点を変換
        transformed_corners = self.transform_points(corners, from_system, to_system)

        # 新しい境界ボックスを計算
        new_min = transformed_corners.min(axis=0)
        new_max = transformed_corners.max(axis=0)
        new_center = (new_min + new_max) / 2
        new_size = new_max - new_min
        new_diagonal = np.linalg.norm(new_size)

        return {
            "min": new_min,
            "max": new_max,
            "center": new_center,
            "size": new_size,
            "diagonal": new_diagonal,
        }

    def create_ground_plane_grid(
        self,
        bbox: Dict[str, Any],
        grid_config: Dict[str, Any],
        coordinate_system: CoordinateSystem = CoordinateSystem.PYDECK,
    ) -> List[Dict[str, Any]]:
        """地面グリッドの生成

        Args:
            bbox: 境界ボックス情報
            grid_config: グリッド設定
            coordinate_system: 座標系

        Returns:
            グリッド線のリスト
        """
        if not grid_config.get("show_xz_grid", True):
            return []

        # 座標系に応じて地面平面を決定
        if coordinate_system == CoordinateSystem.PYDECK:
            # Pydeck座標系では地面はX-Y平面（Z=0）
            x_min, x_max = bbox["min"][0], bbox["max"][0]
            y_min, y_max = bbox["min"][1], bbox["max"][1]
            ground_level = 0  # Z=0

            # グリッドの範囲を拡張
            x_range = (x_max - x_min) * grid_config["padding_ratio"]
            y_range = (y_max - y_min) * grid_config["padding_ratio"]

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            # グリッドの実際の範囲
            grid_x_min = x_center - x_range / 2
            grid_x_max = x_center + x_range / 2
            grid_y_min = y_center - y_range / 2
            grid_y_max = y_center + y_range / 2

            lines = []
            num_lines = grid_config["grid_lines"]

            # X方向の線（Y軸に平行）
            for i in range(num_lines):
                x_pos = grid_x_min + (grid_x_max - grid_x_min) * i / (num_lines - 1)
                lines.append(
                    {
                        "start": [x_pos, grid_y_min, ground_level],
                        "end": [x_pos, grid_y_max, ground_level],
                        "color": grid_config["grid_color"],
                    }
                )

            # Y方向の線（X軸に平行）
            for i in range(num_lines):
                y_pos = grid_y_min + (grid_y_max - grid_y_min) * i / (num_lines - 1)
                lines.append(
                    {
                        "start": [grid_x_min, y_pos, ground_level],
                        "end": [grid_x_max, y_pos, ground_level],
                        "color": grid_config["grid_color"],
                    }
                )

        elif coordinate_system == CoordinateSystem.OPEN3D:
            # Open3D座標系では地面はX-Z平面（Y=0）
            x_min, x_max = bbox["min"][0], bbox["max"][0]
            z_min, z_max = bbox["min"][2], bbox["max"][2]
            ground_level = 0  # Y=0

            # グリッドの範囲を拡張
            x_range = (x_max - x_min) * grid_config["padding_ratio"]
            z_range = (z_max - z_min) * grid_config["padding_ratio"]

            x_center = (x_min + x_max) / 2
            z_center = (z_min + z_max) / 2

            # グリッドの実際の範囲
            grid_x_min = x_center - x_range / 2
            grid_x_max = x_center + x_range / 2
            grid_z_min = z_center - z_range / 2
            grid_z_max = z_center + z_range / 2

            lines = []
            num_lines = grid_config["grid_lines"]

            # X方向の線（Z軸に平行）
            for i in range(num_lines):
                x_pos = grid_x_min + (grid_x_max - grid_x_min) * i / (num_lines - 1)
                lines.append(
                    {
                        "start": [x_pos, ground_level, grid_z_min],
                        "end": [x_pos, ground_level, grid_z_max],
                        "color": grid_config["grid_color"],
                    }
                )

            # Z方向の線（X軸に平行）
            for i in range(num_lines):
                z_pos = grid_z_min + (grid_z_max - grid_z_min) * i / (num_lines - 1)
                lines.append(
                    {
                        "start": [grid_x_min, ground_level, z_pos],
                        "end": [grid_x_max, ground_level, z_pos],
                        "color": grid_config["grid_color"],
                    }
                )
        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate_system}")

        return lines

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
                "ground_plane": "X-Z平面 (Y=0)",
            }
        elif system == CoordinateSystem.PYDECK:
            return {
                "name": "Pydeck",
                "x_axis": "右",
                "y_axis": "上",
                "z_axis": "前方 (-Z画面内)",
                "ground_plane": "X-Y平面 (Z=0)",
            }
        else:
            raise ValueError(f"Unknown coordinate system: {system}")


# グローバルなトランスフォーマーインスタンス
transformer = CoordinateTransformer()

# ===== 設定と定数 =====
POINT_CLOUD_CONFIG = {
    "num_points": 10000,
    "point_size": 2,
}

CAMERA_CONFIG = {
    "rotation_x": 15,
    "rotation_orbit": 30,
    "zoom_padding": 0.05,  # 境界ボックスに対するズームの余白
}

COLOR_CONFIG = {
    "colormap": "viridis",  # 距離ベースのカラーマップ
    "axis_colors": {
        "x": [255, 0, 0],  # 赤
        "y": [0, 255, 0],  # 緑
        "z": [0, 0, 255],  # 青
    },
    "axis_length": 0.1,  # 軸の長さ（境界ボックスに対する比率）
}

GRID_CONFIG = {
    "show_xz_grid": True,
    "grid_lines": 21,  # グリッドの線の数（奇数にすると中心線が引かれる）
    "grid_color": [150, 150, 150, 100],  # グリッド線の色（RGBA）
    "grid_width": 1,  # グリッド線の太さ
    "padding_ratio": 1.2,  # グリッドの広さ（点群サイズに対する比率）
}

BACKGROUND_COLOR = [128, 128, 128]  # グレー背景

# ページ設定
st.set_page_config(layout="wide")
st.title("高性能3D点群ビューア (Pydeck版)")
st.markdown("インタラクティブな3D可視化ライブラリPydeckを使用し、リッチな情報提示を実現します。")


# ===== 点群処理関数 =====
@st.cache_resource
def load_bunny_point_cloud() -> o3d.geometry.PointCloud:
    """Open3Dのサンプルデータから点群を読み込み、中心化する"""
    bunny_mesh_path = o3d.data.BunnyMesh().path
    mesh = o3d.io.read_triangle_mesh(bunny_mesh_path)
    pcd = mesh.sample_points_uniformly(
        number_of_points=POINT_CLOUD_CONFIG["num_points"]
    )
    pcd.translate(-pcd.get_center())
    return pcd


def calculate_bounding_box(points: np.ndarray) -> Dict[str, float]:
    """点群の境界ボックスを計算"""
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2
    size = max_bound - min_bound
    diagonal = np.linalg.norm(size)

    return {
        "min": min_bound,
        "max": max_bound,
        "center": center,
        "size": size,
        "diagonal": diagonal,
    }


def apply_distance_based_colors(points: np.ndarray) -> np.ndarray:
    """カメラ原点からの距離に基づいて色を計算"""
    # 各点の原点からの距離を計算
    distances = np.linalg.norm(points, axis=1)

    # 距離を0-1の範囲に正規化
    min_dist = distances.min()
    max_dist = distances.max()
    normalized_distances = (distances - min_dist) / (max_dist - min_dist + 1e-6)

    # カラーマップを適用
    colormap = plt.get_cmap(COLOR_CONFIG["colormap"])
    colors = colormap(normalized_distances)[:, :3]  # RGBのみ取得（アルファ除外）

    # 0-255の範囲に変換
    return (colors * 255).astype(np.uint8)


def calculate_optimal_camera_settings(bbox: Dict[str, float]) -> Dict[str, Any]:
    """境界ボックスに基づいて最適なカメラ設定を計算"""
    # ズームレベルを対角線の長さに基づいて計算
    # Pydeckのズームは対数スケールなので、適切な値に変換
    zoom = 5.5 - np.log2(bbox["diagonal"] * CAMERA_CONFIG["zoom_padding"])
    zoom_value = max(1.0, min(20.0, zoom))
    print(zoom_value)

    return {
        "target": [0, 0, 0],  # 原点を見る
        "controller": True,
        "rotation_x": CAMERA_CONFIG["rotation_x"],
        "rotation_orbit": CAMERA_CONFIG["rotation_orbit"],
        "zoom": zoom_value,  # ズームを適切な範囲に制限
    }


# ===== レイヤー作成関数 =====
def create_point_cloud_layer(df: pd.DataFrame) -> pdk.Layer:
    """点群レイヤーを作成"""
    return pdk.Layer(
        "PointCloudLayer",
        data=df,
        get_position=["x", "y", "z"],
        get_color=["r", "g", "b"],
        get_normal=[0, 0, 15],
        auto_highlight=True,
        pickable=True,
        point_size=POINT_CLOUD_CONFIG["point_size"],
    )


def create_axis_layers(axis_scale: float) -> list:
    """座標軸レイヤーを作成"""
    layers = []

    # 各軸のデータを作成
    axes_data = {
        "x": {
            "start": [0, 0, 0],
            "end": [axis_scale, 0, 0],
            "color": COLOR_CONFIG["axis_colors"]["x"],
        },
        "y": {
            "start": [0, 0, 0],
            "end": [0, axis_scale, 0],
            "color": COLOR_CONFIG["axis_colors"]["y"],
        },
        "z": {
            "start": [0, 0, 0],
            "end": [0, 0, axis_scale],
            "color": COLOR_CONFIG["axis_colors"]["z"],
        },
    }

    for axis_name, axis_info in axes_data.items():
        df_axis = pd.DataFrame(
            [
                {
                    "start": axis_info["start"],
                    "end": axis_info["end"],
                    "color": axis_info["color"],
                }
            ]
        )

        layer = pdk.Layer(
            "LineLayer",
            data=df_axis,
            get_source_position="start",
            get_target_position="end",
            get_color="color",
            get_width=3,
            pickable=False,
        )
        layers.append(layer)

    return layers


def create_grid_layer(bbox: Dict[str, float]) -> list:
    """地面グリッドレイヤーを作成（座標変換モジュール使用）"""
    lines = transformer.create_ground_plane_grid(
        bbox, GRID_CONFIG, CoordinateSystem.PYDECK
    )

    if not lines:
        return []

    df_grid = pd.DataFrame(lines)

    return [
        pdk.Layer(
            "LineLayer",
            data=df_grid,
            get_source_position="start",
            get_target_position="end",
            get_color="color",
            get_width=GRID_CONFIG["grid_width"],
            pickable=False,
        )
    ]


# ===== メイン処理 =====
def main():
    # 点群データの読み込み
    pcd = load_bunny_point_cloud()
    points_open3d = np.asarray(pcd.points)

    # 座標系変換（Open3D -> Pydeck）
    points = transformer.transform_points(
        points_open3d, CoordinateSystem.OPEN3D, CoordinateSystem.PYDECK
    )

    # 境界ボックスの計算
    bbox = calculate_bounding_box(points)

    # 距離ベースの色を適用
    colors = apply_distance_based_colors(points)

    # DataFrameの作成
    df = pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "r": colors[:, 0],
            "g": colors[:, 1],
            "b": colors[:, 2],
        }
    )

    # レイヤーの作成
    point_cloud_layer = create_point_cloud_layer(df)
    axis_scale = bbox["diagonal"] * COLOR_CONFIG["axis_length"]
    axis_layers = create_axis_layers(axis_scale)
    grid_layers = create_grid_layer(bbox)

    # カメラ設定の計算
    view_state = pdk.ViewState(**calculate_optimal_camera_settings(bbox))

    # ビューの設定
    view = pdk.View(type="OrbitView", controller=True)

    # Deckオブジェクトの作成
    deck = pdk.Deck(
        layers=[point_cloud_layer] + axis_layers + grid_layers,
        initial_view_state=view_state,
        views=[view],
        map_style=None,
    )

    # 表示
    components.html(deck.to_html(as_string=True), height=700, scrolling=False)

    # 操作方法の表示
    st.info(
        """
    **操作方法**
    - **回転:** マウスの左ボタンを押しながらドラッグ
    - **移動:** マウスの右ボタン->左ボタンを押しながらドラッグ
    - **ズーム:** マウスホイールをスクロール
    """
    )
    
    # 再生成ボタン
    if st.button("再生成", type="primary"):
        st.rerun()


if __name__ == "__main__":
    main()

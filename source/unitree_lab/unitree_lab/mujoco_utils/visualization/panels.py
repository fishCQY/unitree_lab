# Copyright (c) unitree_lab contributors.
# SPDX-License-Identifier: Apache-2.0

"""Visualization panels for MuJoCo simulation."""

from __future__ import annotations

import os

import cv2
import numpy as np

# Optional FreeType support for better text rendering
try:
    from cv2 import freetype  # type: ignore

    _HAS_FREETYPE = hasattr(cv2, "freetype")
except Exception:
    freetype = None  # type: ignore
    _HAS_FREETYPE = False

_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]
_FONT_PATH = next((p for p in _FONT_SEARCH_PATHS if os.path.exists(p)), None)
_FT2 = None


# =============================================================================
# Exteroception Visualization Panel
# =============================================================================


def create_exteroception_visualization(
    exteroception: np.ndarray,
    height: int,
    width: int,
    exteroception_type: str | None = None,
    robot_z: float = 0.0,
) -> np.ndarray:
    """Create a visualization panel showing exteroception as a heatmap.

    Args:
        exteroception: Exteroception data in image format (grid_h, grid_w, channels) or (grid_h, grid_w).
        height: Output image height.
        width: Output image width.
        exteroception_type: Type of exteroception ("height_scan", "depth", etc.) for title.
        robot_z: Robot base Z position for display (optional).

    Returns:
        BGR visualization image.
    """
    # Remove channel dimension if present
    if exteroception.ndim == 3:
        viz_data = exteroception[:, :, 0]
    else:
        viz_data = exteroception

    if exteroception_type == "height_scan":
        viz_data = -viz_data

    grid_h, grid_w = viz_data.shape

    # Layout constants
    TOP_MARGIN, BOTTOM_MARGIN = 50, 40
    LEFT_MARGIN, RIGHT_MARGIN = 10, 10

    # Create black background
    viz_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Add title based on type
    if exteroception_type == "height_scan":
        title = f"Height Scan ({grid_h}x{grid_w})"
    elif exteroception_type == "depth":
        title = f"Depth Scan ({grid_h}x{grid_w})"
    else:
        title = f"Exteroception ({grid_h}x{grid_w})"

    cv2.putText(viz_img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show robot Z if provided
    if robot_z != 0.0:
        cv2.putText(viz_img, f"Robot Z: {robot_z:.2f}m", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Calculate heatmap area preserving aspect ratio for depth images
    available_w = width - LEFT_MARGIN - RIGHT_MARGIN
    available_h = height - TOP_MARGIN - BOTTOM_MARGIN

    if exteroception_type == "depth" and grid_h > 0 and grid_w > 0:
        data_aspect = grid_w / grid_h  # e.g. 48/30 = 1.6
        avail_aspect = available_w / available_h
        if data_aspect > avail_aspect:
            heatmap_w = available_w
            heatmap_h = int(available_w / data_aspect)
        else:
            heatmap_h = available_h
            heatmap_w = int(available_h * data_aspect)
    else:
        heatmap_w = available_w
        heatmap_h = available_h

    # Normalize to [0, 255] for visualization
    vmin, vmax = viz_data.min(), viz_data.max()
    if exteroception_type == "height_scan":
        vmin, vmax = min(-max(abs(vmin), abs(vmax)), -0.2), max(max(abs(vmin), abs(vmax)), 0.2)

    if vmax - vmin > 0:
        normalized = ((viz_data - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(viz_data, dtype=np.uint8)

    # For height_scan, flip both axes so front-of-robot is at top.
    # For depth camera images, the raw orientation is already correct
    # (top = far/forward, bottom = near/ground), so no flip is needed.
    if exteroception_type != "depth":
        normalized = np.flip(normalized, axis=(0, 1))
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    # Resize heatmap to fit the designated area
    heatmap_resized = cv2.resize(heatmap, (heatmap_w, heatmap_h), interpolation=cv2.INTER_NEAREST)

    # Center heatmap in available area
    y_offset = TOP_MARGIN + (available_h - heatmap_h) // 2
    x_offset = LEFT_MARGIN + (available_w - heatmap_w) // 2

    # Place heatmap in visualization
    viz_img[y_offset : y_offset + heatmap_h, x_offset : x_offset + heatmap_w] = heatmap_resized

    # Draw grid lines
    cell_w, cell_h = heatmap_w / grid_w, heatmap_h / grid_h
    for i in range(grid_h + 1):
        y = int(y_offset + i * cell_h)
        cv2.line(viz_img, (x_offset, y), (x_offset + heatmap_w, y), (50, 50, 50), 1)
    for j in range(grid_w + 1):
        x = int(x_offset + j * cell_w)
        cv2.line(viz_img, (x, y_offset), (x, y_offset + heatmap_h), (50, 50, 50), 1)

    # Draw robot marker at grid center with forward arrow
    center_x = int(x_offset + (grid_w / 2) * cell_w)
    center_y = int(y_offset + (grid_h / 2) * cell_h)
    cv2.circle(viz_img, (center_x, center_y), 5, (255, 255, 255), -1)
    cv2.circle(viz_img, (center_x, center_y), 5, (0, 0, 0), 1)
    cv2.arrowedLine(
        viz_img, (center_x, center_y), (center_x, center_y - int(2 * cell_h)), (255, 255, 255), 2, tipLength=0.3
    )

    # Draw color bar legend
    legend_y = height - 25
    legend_h = 15
    gradient = np.linspace(0, 255, heatmap_w).astype(np.uint8).reshape(1, -1)
    legend_bar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    legend_bar = cv2.resize(legend_bar, (heatmap_w, legend_h))
    viz_img[legend_y : legend_y + legend_h, LEFT_MARGIN : LEFT_MARGIN + heatmap_w] = legend_bar

    # Legend labels
    cv2.putText(
        viz_img,
        f"Low ({vmin:.2f})",
        (LEFT_MARGIN, legend_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 100, 100),
        1,
    )
    cv2.putText(
        viz_img,
        f"High ({vmax:.2f})",
        (LEFT_MARGIN + heatmap_w - 75, legend_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (150, 150, 255),
        1,
    )

    return viz_img


# =============================================================================
# Torque Visualization Panel
# =============================================================================


def create_torque_visualization(
    torque_util: np.ndarray,
    width: int,
    height: int,
    num_actions: int,
    base_linear_vel: np.ndarray | None = None,
) -> np.ndarray:
    """Create a visualization panel showing motor torque utilization.

    Args:
        torque_util: Motor torque utilization (0 to 1 array).
        width: Image width.
        height: Image height.
        num_actions: Number of motors/actions.
        base_linear_vel: Base linear velocity [vx, vy, vz] (optional).

    Returns:
        BGR visualization image.
    """
    # Create black background
    viz_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Define margins for title and motor ID labels
    top_margin = 50
    bottom_margin = 25

    # Add title
    title = "Torque Usage (%)"
    cv2.putText(viz_img, title, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display base linear velocity
    if base_linear_vel is not None:
        vel_magnitude = np.linalg.norm(base_linear_vel[:2])
        vel_text = f"Vel: {vel_magnitude:.2f} m/s"
        vel_detail_text = f"vx: {base_linear_vel[0]:.2f}, vy: {base_linear_vel[1]:.2f}"

        font_scale = 0.6
        font_thickness = 1
        vel_text_size, _ = cv2.getTextSize(vel_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        vel_detail_text_size, _ = cv2.getTextSize(vel_detail_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        cv2.putText(
            viz_img,
            vel_text,
            (width - vel_text_size[0] - 5, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            font_thickness,
        )
        cv2.putText(
            viz_img,
            vel_detail_text,
            (width - vel_detail_text_size[0] - 5, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

    # Calculate bar dimensions
    bar_total_width = width / num_actions
    bar_gap = max(1, int(bar_total_width * 0.1))
    bar_drawable_width = int(bar_total_width - bar_gap)

    # Draw bars for each motor
    for i in range(num_actions):
        util = torque_util[i]

        # Color based on utilization (BGR format)
        if util > 0.9:
            color = (0, 0, 255)  # Red
        elif util > 0.75:
            color = (0, 255, 255)  # Yellow
        elif util > 0.5:
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 255, 0)  # Green

        # Calculate bar dimensions and position
        bar_max_height = height - top_margin - bottom_margin
        bar_height = int(util * bar_max_height)
        x1 = int(i * bar_total_width)
        y2 = height - bottom_margin
        y1 = y2 - bar_height
        x2 = x1 + bar_drawable_width

        # Draw filled rectangle
        cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, -1)

        # Add motor ID text below bar
        motor_id_text = str(i)
        font_scale = 0.4
        font_thickness = 1

        text_size, _ = cv2.getTextSize(motor_id_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x = x1 + (bar_drawable_width - text_size[0]) // 2
        text_y = height - 8

        cv2.putText(
            viz_img,
            motor_id_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )

    return viz_img


# =============================================================================
# Command Overlay Helpers
# =============================================================================


def _get_ft2():
    """Lazily create a FreeType renderer if available."""
    global _FT2
    if _FT2 is not None:
        return _FT2
    if not (_HAS_FREETYPE and _FONT_PATH):
        return None
    try:
        ft2 = freetype.createFreeType2()
        ft2.loadFontData(fontFileName=_FONT_PATH, id=0)
        _FT2 = ft2
        return ft2
    except Exception:
        return None


def _put_text(img, text, org, font_scale, color, thickness):
    """Draw text with FreeType if available; otherwise fallback to Hershey."""
    ft2 = _get_ft2()
    if ft2 is None:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return
    font_height = max(16, int(32 * font_scale))
    ft2.putText(
        img,
        text,
        org,
        fontHeight=font_height,
        color=color,
        thickness=thickness,
        line_type=cv2.LINE_AA,
        bottomLeftOrigin=False,
    )


def _command_icons(command: np.ndarray) -> tuple[str, str, str]:
    """Map velocity commands to icons; prefer Unicode arrows if font is ready."""
    use_unicode = _get_ft2() is not None

    def pick_icon(val: float, pos_icon: str, neg_icon: str) -> str:
        if abs(val) < 1e-3:
            return "⏸" if use_unicode else "HOLD"
        return pos_icon if val > 0 else neg_icon

    vx, vy, yaw = command.tolist()
    if use_unicode:
        return pick_icon(vx, "↑", "↓"), pick_icon(vy, "→", "←"), pick_icon(yaw, "⟳", "⟲")
    return pick_icon(vx, "FWD", "BACK"), pick_icon(vy, "RIGHT", "LEFT"), pick_icon(yaw, "CCW", "CW")


def _draw_command_overlay(
    frame: np.ndarray,
    command: np.ndarray | None,
    command_idx: int | None,
    sim_time: float,
    overlay_width: int | None = None,
) -> np.ndarray:
    """Overlay command sequence, icons, numeric values, and time on the frame."""
    if command is None:
        return frame

    overlay = frame.copy()
    bar_height = 80
    overlay_w = overlay_width or frame.shape[1]
    overlay_w = min(overlay_w, frame.shape[1])

    # Semi-transparent bar
    cv2.rectangle(overlay, (0, 0), (overlay_w, bar_height), (0, 0, 0), -1)
    alpha = 0.45
    frame[:bar_height, :overlay_w] = cv2.addWeighted(
        overlay[:bar_height, :overlay_w],
        alpha,
        frame[:bar_height, :overlay_w],
        1 - alpha,
        0,
    )

    icons = _command_icons(command)
    seq_val = command_idx if command_idx is not None else "-"
    time_text = f"Time: {sim_time:.2f}s"
    icon_text = f"{icons[0]}  {icons[1]}  {icons[2]}"
    seq_cmd_text = f"Seq: {seq_val}   cmd vx={command[0]:.2f}, vy={command[1]:.2f}, yaw={command[2]:.2f}"

    # First row: time + icons
    _put_text(frame, time_text, (10, 28), 0.8, (255, 255, 255), 2)
    _put_text(frame, icon_text, (200, 28), 0.8, (0, 255, 255), 2)

    # Second row: seq + cmd
    _put_text(frame, seq_cmd_text, (10, 60), 0.65, (210, 210, 210), 1)

    return frame


# =============================================================================
# Combined Visualization Frame
# =============================================================================


def create_combined_visualization(
    sim_frame: np.ndarray,
    torque_util: np.ndarray,
    num_actions: int,
    base_lin_vel: np.ndarray,
    exteroception: np.ndarray | None = None,
    exteroception_type: str | None = None,
    robot_z: float = 0.0,
    sim_time: float = 0.0,
    command: np.ndarray | None = None,
    command_idx: int | None = None,
) -> np.ndarray:
    """Create a combined visualization frame with simulation view, torque panel, and optional exteroception.

    Layout:
    - If exteroception is provided:
        +------------------+---------------+----------+
        |                  | Exteroception |  Torque  |
        |   Simulation     |    (e.g.,     |  Usage   |
        |                  |  Height/Depth)|          |
        +------------------+---------------+----------+
    - If exteroception is None:
        +------------------+------------------+
        |   Simulation     |  Torque Usage    |
        +------------------+------------------+

    Args:
        sim_frame: RGB simulation frame from MuJoCo renderer.
        torque_util: Motor torque utilization (0 to 1 array).
        num_actions: Number of motors/actions.
        base_lin_vel: Base linear velocity [vx, vy, vz] in robot frame.
        exteroception: Exteroception data in image format (height, width, channels), or None.
        exteroception_type: Type of exteroception for display title.
        robot_z: Robot base Z position for display.
        sim_time: Current simulation time for overlay.
        command: Velocity command [vx, vy, yaw] for overlay.
        command_idx: Command sequence index for overlay.

    Returns:
        Combined RGB visualization frame.
    """
    sim_height, sim_width, _ = sim_frame.shape

    if exteroception is not None:
        # Layout with exteroception: Left=Sim, Middle=Exteroception, Right=Torque
        panel_width = sim_width // 2
        panel_height = sim_height

        # Create exteroception visualization (middle)
        ext_viz_bgr = create_exteroception_visualization(
            exteroception, panel_height, panel_width, exteroception_type, robot_z
        )
        ext_viz_rgb = cv2.cvtColor(ext_viz_bgr, cv2.COLOR_BGR2RGB)

        # Create torque visualization (right)
        torque_viz_bgr = create_torque_visualization(torque_util, panel_width, panel_height, num_actions, base_lin_vel)
        torque_viz_rgb = cv2.cvtColor(torque_viz_bgr, cv2.COLOR_BGR2RGB)

        # Stack right panels horizontally
        right_panel = np.concatenate([ext_viz_rgb, torque_viz_rgb], axis=1)

        # Combine left (sim) and right panels horizontally
        combined_frame = np.concatenate([sim_frame, right_panel], axis=1)
    else:
        # Original layout without exteroception: [Simulation | Torque]
        torque_viz_bgr = create_torque_visualization(torque_util, sim_width, sim_height, num_actions, base_lin_vel)
        torque_viz_rgb = cv2.cvtColor(torque_viz_bgr, cv2.COLOR_BGR2RGB)

        combined_frame = np.concatenate([sim_frame, torque_viz_rgb], axis=1)

    # Add command + time overlay (restrict to simulation area)
    combined_frame = _draw_command_overlay(combined_frame, command, command_idx, sim_time, overlay_width=sim_width)

    return combined_frame

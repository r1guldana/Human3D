# viewer.py
import numpy as np
import trimesh
import time
from pathlib import Path

OUTPUT_DIR   = Path("data/output")
FPS          = 30
FRAME_DELAY  = 1.0 / FPS


def load_data():

    all_vertices = np.load(OUTPUT_DIR / "all_vertices.npy")  # (N, 10475, 3)
    faces        = np.load(OUTPUT_DIR / "faces.npy")         # (20908, 3)

    print(f"Frames   : {len(all_vertices)}")
    print(f"Vertices : {all_vertices.shape[1]}")
    print(f"Faces    : {len(faces)}")

    return all_vertices, faces


def center_vertices(all_vertices):
    """
    Центрировать все кадры по первому кадру.
    Убирает смещение по X/Z, оставляет высоту.
    """

    origin = all_vertices[0].mean(axis=0)
    origin[1] = 0  # не трогаем высоту Y

    return all_vertices - origin


def animate(all_vertices, faces, fps=FPS):

    frame_delay = 1.0 / fps
    n_frames    = len(all_vertices)

    print(f"\nStarting animation: {n_frames} frames at {fps} FPS")
    print("Controls:")
    print("  Mouse drag — rotate")
    print("  Scroll     — zoom")
    print("  Q          — quit")

    # Создаём начальный меш
    mesh  = trimesh.Trimesh(
        vertices=all_vertices[0],
        faces=faces,
        process=False
    )

    # Цвет — телесный
    mesh.visual.vertex_colors = [200, 160, 130, 255]

    scene = mesh.scene()

    # Открыть окно
    window = scene.show(
        smooth=True,
        flags={"cull": False}
    )

def animateOpen3d(all_vertices, faces, fps=FPS):

    import open3d as o3d

    n_frames    = len(all_vertices)
    frame_delay = 1.0 / fps

    print(f"Frames: {n_frames} | FPS: {fps}")
    print("Close window to stop")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="3D Motion Replay",
        width=800,
        height=600
    )

    # Создаём меш первого кадра
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(all_vertices[0])
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.6, 0.5])

    vis.add_geometry(mesh)

    # Зафиксировать камеру по первому кадру
    vis.reset_view_point(True)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.light_on            = True

    frame_id   = 0
    last_time  = time.time()

    while True:

        now = time.time()

        # Обновляем только если прошло достаточно времени
        if now - last_time >= frame_delay:

            # ── Ключевое исправление ──────────────────────
            # Нужно remove → обновить → add заново
            vis.remove_geometry(mesh, reset_bounding_box=False)

            mesh.vertices = o3d.utility.Vector3dVector(
                all_vertices[frame_id]
            )
            mesh.compute_vertex_normals()

            vis.add_geometry(mesh, reset_bounding_box=False)
            # ──────────────────────────────────────────────

            frame_id  = (frame_id + 1) % n_frames
            last_time = now

            print(f"\r  Frame {frame_id:4d}/{n_frames}", end="", flush=True)

        vis.poll_events()
        vis.update_renderer()

        # Окно закрыто
        if not vis.poll_events():
            break

    print()
    vis.destroy_window()

def animate_export_gif(all_vertices, faces, output_path, fps=15):
    """
    Экспорт анимации в GIF.
    pip install imageio pillow
    """

    import imageio
    from PIL import Image

    print(f"Exporting GIF: {len(all_vertices)} frames...")

    images = []

    for i, verts in enumerate(all_vertices):

        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            process=False
        )
        mesh.visual.vertex_colors = [200, 160, 130, 255]

        scene  = mesh.scene()

        # Рендерить кадр
        png    = scene.save_image(resolution=(512, 512))
        image  = Image.open(
            __import__("io").BytesIO(png)
        )
        images.append(np.array(image))

        if i % 10 == 0:
            print(f"  Frame {i+1}/{len(all_vertices)}")

    imageio.mimsave(output_path, images, fps=fps)
    print(f"Saved: {output_path}")


def animate_export_video(all_vertices, faces, output_path, fps=30):
    """
    Экспорт анимации в MP4.
    pip install imageio[ffmpeg]
    """

    import imageio
    import io
    from PIL import Image

    print(f"Exporting MP4: {len(all_vertices)} frames...")

    writer = imageio.get_writer(output_path, fps=fps)

    for i, verts in enumerate(all_vertices):

        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            process=False
        )
        mesh.visual.vertex_colors = [200, 160, 130, 255]

        scene = mesh.scene()
        png   = scene.save_image(resolution=(512, 512))
        image = np.array(
            Image.open(io.BytesIO(png))
        )

        writer.append_data(image)

        if i % 10 == 0:
            print(f"  Frame {i+1}/{len(all_vertices)}")

    writer.close()
    print(f"Saved: {output_path}")

def animate_matplotlib(all_vertices, faces, fps=FPS):

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    print(f"Frames: {len(all_vertices)} | FPS: {fps}")

    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # Берём каждый 3й кадр для скорости matplotlib
    step   = 3
    frames = all_vertices[::step]

    # Вычислить границы сцены по всем кадрам
    all_pts = all_vertices.reshape(-1, 3)
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()

    def update(frame_id):

        ax.clear()

        verts = frames[frame_id]

        # Рисуем как облако точек (быстро)
        ax.scatter(
            verts[:, 0],
            verts[:, 2],   # Z вверх в matplotlib
            verts[:, 1],
            s=0.3,
            c=verts[:, 1],  # цвет по высоте
            cmap="RdYlBu",
            alpha=0.6
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_zlim(y_min, y_max)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        ax.set_title(f"Frame {frame_id * step + 1}/{len(all_vertices)}")

        # Убрать оси для чистого вида
        ax.grid(False)
        ax.set_axis_off()

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False
    )

    plt.tight_layout()
    plt.show()

    return anim

if __name__ == "__main__":

    all_vertices, faces = load_data()

    all_vertices = center_vertices(all_vertices)

    #animate(all_vertices, faces)
    animate_matplotlib(
            all_vertices,
            faces,
            fps=30
        )


"""
    elif choice == "2":
        animate_export_gif(
            all_vertices,
            faces,
            output_path=str(OUTPUT_DIR / "animation.gif"),
            fps=15
        )

    elif choice == "3":
        animate_export_video(
            all_vertices,
            faces,
            output_path=str(OUTPUT_DIR / "animation.mp4"),
            fps=30
        )

    else:
        print("Invalid choice")"""
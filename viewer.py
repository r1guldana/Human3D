import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Загрузка
# -----------------------------

vertices = np.load("data/output/vertices.npy")
joints = np.load("data/output/joints.npy")

print("Vertices:", vertices.shape)
print("Joints:", joints.shape)

# -----------------------------
# Отрисовка
# -----------------------------

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Вершины тела
ax.scatter(
    vertices[:, 0],
    vertices[:, 1],
    vertices[:, 2],
    s=0.2,
    alpha=0.25,
    label="Vertices"
)

# Суставы
ax.scatter(
    joints[:, 0],
    joints[:, 1],
    joints[:, 2],
    s=30,
    c="red",
    label="Joints"
)

# Подписать номера суставов
for i, p in enumerate(joints):
    ax.text(
        p[0],
        p[1],
        p[2],
        str(i),
        fontsize=8,
        color="blue"
    )

# -----------------------------
# Одинаковый масштаб осей
# -----------------------------

all_points = np.vstack((vertices, joints))

xmin, ymin, zmin = all_points.min(axis=0)
xmax, ymax, zmax = all_points.max(axis=0)

center = np.array([
    (xmin + xmax) / 2,
    (ymin + ymax) / 2,
    (zmin + zmax) / 2
])

radius = max(
    xmax - xmin,
    ymax - ymin,
    zmax - zmin
) / 2

ax.set_xlim(center[0] - radius, center[0] + radius)
ax.set_ylim(center[1] - radius, center[1] + radius)
ax.set_zlim(center[2] - radius, center[2] + radius)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.legend()

plt.tight_layout()
plt.show()
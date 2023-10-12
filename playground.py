import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from euler_to_rot import get_rotation_matrix


pts = np.load("playground_tmp/pts.npy")
print(f"pts {pts.shape}")
pose = np.load("playground_tmp/pose_0.npy")
yaw, pitch, roll = pose

if len(pts.shape) > 2:
    first_pts = pts[0]

rot_mat = get_rotation_matrix(yaw, pitch, roll)
new_points = np.dot(rot_mat, first_pts)

nose_idx = 30
left_ear_idx = 0
right_ear_idx = 16

norm_points = new_points - new_points[:, nose_idx].reshape(3, 1)
max_x = max(np.abs(norm_points[0, left_ear_idx]), np.abs(norm_points[0, right_ear_idx]))
print(max_x)
norm_points = norm_points / max_x

back_points = norm_points * max_x
back_points = back_points + new_points[:, nose_idx].reshape(3, 1)
back_points = np.dot(np.linalg.inv(rot_mat), back_points)


img = cv2.imread("vfhq_sq_img.png")
height, width = img.shape[:2]
plt.figure(figsize=(12, height / width * 12))
plt.imshow(img[..., ::-1])
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.axis("off")

alpha = 0.8
markersize = 4
lw = 1.5
color = "w"
markeredgecolor = "black"

nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]
i = 0
# close eyes and mouths
plot_close = lambda i1, i2: plt.plot(
    [pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]], color=color, lw=lw, alpha=alpha - 0.1
)
plot_close(41, 36)
plot_close(47, 42)
plot_close(59, 48)
plot_close(67, 60)

for ind in range(len(nums) - 1):
    l, r = nums[ind], nums[ind + 1]
    plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

    plt.plot(
        pts[i][0, l:r],
        pts[i][1, l:r],
        marker="o",
        linestyle="None",
        markersize=markersize,
        color=color,
        markeredgecolor=markeredgecolor,
        alpha=alpha,
    )

plt.savefig("playground_tmp/raw_points.png")
plt.close()


plot_close = lambda i1, i2: plt.plot(
    [norm_points[0, i1], norm_points[0, i2]],
    [norm_points[1, i1], norm_points[1, i2]],
    color=color,
    lw=lw,
    alpha=alpha - 0.1,
)
plot_close(41, 36)
plot_close(47, 42)
plot_close(59, 48)
plot_close(67, 60)

for ind in range(len(nums) - 1):
    l, r = nums[ind], nums[ind + 1]
    plt.plot(norm_points[0, l:r], norm_points[1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

    plt.plot(
        norm_points[0, l:r],
        norm_points[1, l:r],
        marker="o",
        linestyle="None",
        markersize=markersize,
        color=color,
        markeredgecolor=markeredgecolor,
        alpha=alpha,
    )

plt.gca().invert_yaxis()
plt.savefig("playground_tmp/front_points.png")
plt.close()

plt.figure(figsize=(12, height / width * 12))
plt.imshow(img[..., ::-1])
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

plot_close = lambda i1, i2: plt.plot(
    [back_points[0, i1], back_points[0, i2]],
    [back_points[1, i1], back_points[1, i2]],
    color=color,
    lw=lw,
    alpha=alpha - 0.1,
)
plot_close(41, 36)
plot_close(47, 42)
plot_close(59, 48)
plot_close(67, 60)

for ind in range(len(nums) - 1):
    l, r = nums[ind], nums[ind + 1]
    plt.plot(back_points[0, l:r], back_points[1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

    plt.plot(
        back_points[0, l:r],
        back_points[1, l:r],
        marker="o",
        linestyle="None",
        markersize=markersize,
        color=color,
        markeredgecolor=markeredgecolor,
        alpha=alpha,
    )

plt.savefig("playground_tmp/back_points.png")
plt.close()

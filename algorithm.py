# -*- coding: utf-8 -*-
"""
중앙선(흰색) 우선, 없으면 도로 메디얼-스켈레톤 사용
+ 체증 가중치
+ 결과 2개 (dual / allwhite)
+ 직선(도로 관통선) 없음
+ ✅ 마우스 없이 코드로 S/D/체증/다중목표 지정 가능 (USE_MOUSE=False)
+ ✅ 1단계: 목표 3개(a,b,c) 중 '경로가 존재'하고 '총비용 최소'인 곳으로 자동 선택
"""

import cv2, numpy as np
from heapq import heappush, heappop

# =========================
# 이미지/입력 설정
# =========================
img_path = "main.png"

USE_MOUSE = False
START = (70, 410)   # (x, y)

# ▶▶ 목표 3개 (a,b,c)
GOALS = {
    "a": (50, 50),
    "b": (270, 120),
    "c": (650, 120),
}
# ▶▶ 현재 수용 가능 여부 (다음 단계에서 터미널로 토글 예정)
AVAILABLE = {"a": True, "b": True, "c": True}

# ▶▶ 체증 영역(초기 상태). 다음 단계에서 터미널로 on/off 토글 추가 예정
PRESET_BLOCKS = [
    # (300, 300, 500, 450),
]

# -------------------------
# 밝기/마스크 파라미터
# -------------------------
T_GRAY = 20
STRICT_WHITE = 245
CENTERLINE_DILATE = 0

ROAD_ERODE_ITERS = 0
ROAD_OPEN_ITERS  = 1

ALLOW_DIAGONAL = False

# -------------------------
# 체증 비용 파라미터
# -------------------------
CONGESTION_W   = 12.0
ENTRY_PENALTY  = 30.0
INSIDE_PENALTY = 2.5
DILATE_CONGEST = 1

NEI4 = [(0,-1),(0,1),(-1,0),(1,0)]
NEI8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# =========================
# 유틸: 스켈레톤(1px)
# =========================
def skeletonize01(bin01):
    try:
        import cv2.ximgproc as xi
        sk = xi.thinning((bin01 * 255).astype(np.uint8), xi.THINNING_ZHANGSUEN)
        return (sk > 0).astype(np.uint8)
    except Exception:
        prev = np.zeros_like(bin01, np.uint8)
        skel = bin01.copy().astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        while True:
            eroded = cv2.erode(skel, kernel, 1)
            temp = cv2.dilate(eroded, kernel, 1)
            temp = cv2.subtract(skel, temp)
            prev = cv2.bitwise_or(prev, temp)
            skel = eroded
            if not skel.any():
                break
        return (prev > 0).astype(np.uint8)

def medial_axis_center(road01):
    road_u8 = (road01.astype(np.uint8) * 255)
    dist = cv2.distanceTransform((road_u8 > 0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_f = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    maxmap = cv2.dilate(dist_f, kernel, 1)
    ridge = (dist_f == maxmap) & (road01 > 0)
    ridge = ridge.astype(np.uint8)
    try:
        import cv2.ximgproc as xi
        ridge = xi.thinning(ridge * 255, xi.THINNING_ZHANGSUEN)
        ridge = (ridge > 0).astype(np.uint8)
    except Exception:
        ridge = skeletonize01(ridge)
    return ridge

def make_road_and_center(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    road_gray = (gray >= T_GRAY).astype(np.uint8)

    if ROAD_OPEN_ITERS > 0:
        kern = np.ones((3, 3), np.uint8)
        road_gray = cv2.morphologyEx(road_gray, cv2.MORPH_OPEN, kern, iterations=ROAD_OPEN_ITERS)
    if ROAD_ERODE_ITERS > 0:
        road_gray = cv2.erode(road_gray, np.ones((3, 3), np.uint8), iterations=ROAD_ERODE_ITERS)

    b, g, r = cv2.split(img_bgr)
    center_white = ((b >= STRICT_WHITE) & (g >= STRICT_WHITE) & (r >= STRICT_WHITE)).astype(np.uint8)
    center_white &= road_gray
    if CENTERLINE_DILATE > 0:
        k = 2 * CENTERLINE_DILATE + 1
        center_white = cv2.dilate(center_white, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1)

    fallback_center = medial_axis_center(road_gray)
    n_white = int(center_white.sum())
    use_white = (n_white > 0)
    center = center_white if use_white else fallback_center
    return road_gray, center, center_white, fallback_center, use_white

# =========================
# 중앙선 전용 A*  (경로+비용)
# =========================
def astar_on_centerline(center01, start_xy, goal_xy, cong_mask=None,
                        allow_diag=False,
                        congestion_w=12.0, entry_penalty=30.0, inside_penalty=2.5):
    H, W = center01.shape
    sx, sy = start_xy
    gx, gy = goal_xy

    ys, xs = np.where(center01 > 0)
    if len(xs) == 0:
        return [], float("inf")

    def snap_to_center(x, y):
        i = int(np.argmin((xs - x)**2 + (ys - y)**2))
        return int(xs[i]), int(ys[i])

    sx, sy = snap_to_center(sx, sy)
    gx, gy = snap_to_center(gx, gy)

    NEI = NEI8 if allow_diag else NEI4

    def heur(x, y):
        return abs(x - gx) + abs(y - gy) if not allow_diag else ((x - gx)**2 + (y - gy)**2)**0.5

    openh = []
    heappush(openh, (heur(sx, sy), 0.0, (sx, sy)))
    came = {}
    gscore = {(sx, sy): 0.0}
    visited = set()

    while openh:
        f, gcost, (x, y) = heappop(openh)
        if (x, y) in visited:
            continue
        visited.add((x, y))
        if (x, y) == (gx, gy):
            break

        for dx, dy in NEI:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if center01[ny, nx] == 0:
                continue

            step = 1.0 if (dx == 0 or dy == 0) else 1.41421356

            add = 0.0
            if cong_mask is not None:
                a = cong_mask[y, x] > 0
                b = cong_mask[ny, nx] > 0
                if (not a) and b:
                    add += entry_penalty
                if b:
                    ng = gcost + step * congestion_w + inside_penalty + add
                else:
                    ng = gcost + step + add
            else:
                ng = gcost + step

            if ng < gscore.get((nx, ny), float("inf")):
                gscore[(nx, ny)] = ng
                came[(nx, ny)] = (x, y)
                heappush(openh, (ng + heur(nx, ny), ng, (nx, ny)))

    # 경로/비용 복원
    if (gx, gy) not in came and (gx, gy) != (sx, sy):
        return [], float("inf")
    path = []
    cur = (gx, gy)
    while cur is not None:
        path.append((cur[1], cur[0]))
        if cur == (sx, sy):
            break
        cur = came.get(cur)
    path.reverse()
    total_cost = gscore.get((gx, gy), float("inf"))
    return path, total_cost

# =========================
# 마우스 UI (옵션)
# =========================
class Picker:
    def __init__(self, img):
        self.img0 = img.copy()
        self.img = img.copy()
        self.S = None
        self.D = None
        self.window_name = "Centerline / Skeleton Pathfinder"
        self.drawing_block_mode = False
        self.block_p1 = None
        self.blocks = []

    def on_mouse(self, e, x, y, flags, param):
        if self.drawing_block_mode:
            if e == cv2.EVENT_LBUTTONDOWN:
                if self.block_p1 is None:
                    self.block_p1 = (x, y)
                else:
                    x1, y1 = self.block_p1
                    x2, y2 = x, y
                    nx1, ny1 = min(x1, x2), min(y1, y2)
                    nx2, ny2 = max(x1, x2), max(y1, y2)
                    self.blocks.append((nx1, ny1, nx2, ny2))
                    self.block_p1 = None
                self.redraw()
        else:
            if e == cv2.EVENT_LBUTTONDOWN:
                if self.S is None:
                    self.S = (x, y)
                elif self.D is None:
                    self.D = (x, y)
                self.redraw()

    def set_mode(self, is_block_mode):
        self.drawing_block_mode = is_block_mode
        self.block_p1 = None
        self.redraw()

    def clear_blocks(self):
        self.blocks = []
        self.block_p1 = None
        self.redraw()

    def redraw(self):
        self.img = self.img0.copy()
        if self.S is not None:
            cv2.circle(self.img, self.S, 6, (255, 0, 0), -1)
        if self.D is not None:
            cv2.circle(self.img, self.D, 6, (0, 255, 0), -1)
        for (x1, y1, x2, y2) in self.blocks:
            cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if self.block_p1 is not None:
            cv2.circle(self.img, self.block_p1, 5, (0, 0, 255), -1)

        txt = "Mode: BLOCK (R: Switch, C: Clear)" if self.drawing_block_mode else "Mode: S/D (R: Switch)"
        color = (0, 255, 255) if self.drawing_block_mode else (255, 255, 0)
        cv2.putText(self.img, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow(self.window_name, self.img)

# ==== 경로 그리기(센터 위 픽셀만) ====
def draw_path_on_canvas(canvas, center01, path, color=(0,0,255), thick=3, tol=0):
    if len(path) < 2:
        return
    h, w = center01.shape
    path_mask = np.zeros((h, w), np.uint8)
    for (y1, x1), (y2, x2) in zip(path[:-1], path[1:]):
        cv2.line(path_mask, (x1, y1), (x2, y2), 255, thickness=thick, lineType=cv2.LINE_8)
    if tol > 0:
        k = 2*tol + 1
        center_thick = cv2.dilate(center01, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1)
    else:
        center_thick = center01
    masked = cv2.bitwise_and(path_mask, (center_thick * 255).astype(np.uint8))
    ys, xs = np.where(masked > 0)
    canvas[ys, xs] = color

# =========================
# 메인
# =========================
if __name__ == "__main__":
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

    road_gray, center, center_white, center_fallback, used_white = make_road_and_center(img)

    # 입력 방식
    if USE_MOUSE:
        picker = Picker(img)
        cv2.namedWindow(picker.window_name)
        cv2.setMouseCallback(picker.window_name, picker.on_mouse)
        picker.redraw()
        while True:
            k = cv2.waitKey(20)
            if k in (13, 10):
                if picker.S and picker.D:
                    break
            elif k == 27:
                cv2.destroyAllWindows()
                raise SystemExit
            elif k == ord('r'):
                picker.set_mode(not picker.drawing_block_mode)
            elif k == ord('c'):
                picker.clear_blocks()
        S = picker.S
        blocks = picker.blocks
        cv2.destroyWindow(picker.window_name)
    else:
        S = START
        blocks = PRESET_BLOCKS

    # 체증 마스크 (초기 상태)
    H, W = center.shape
    cong_mask = np.zeros((H, W), np.uint8)
    for (x1, y1, x2, y2) in blocks:
        cong_mask[y1:y2, x1:x2] = 1
    if DILATE_CONGEST > 0:
        k = 2*DILATE_CONGEST + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        cong_mask = cv2.dilate(cong_mask, kernel, 1)

    # ▶▶ 여러 목표 중 '가능(AVAILABLE=True)' + '경로 존재' + '총비용 최소' 선택
    best_label, best_goal, best_path, best_cost = None, None, [], float("inf")
    for label, goal in GOALS.items():
        if not AVAILABLE.get(label, True):
            continue  # 수용불가는 스킵
        path, cost = astar_on_centerline(
            center, S, goal,
            cong_mask=cong_mask,
            allow_diag=ALLOW_DIAGONAL,
            congestion_w=CONGESTION_W,
            entry_penalty=ENTRY_PENALTY,
            inside_penalty=INSIDE_PENALTY
        )
        if path and cost < best_cost:
            best_label, best_goal, best_path, best_cost = label, goal, path, cost

    # ===== 시각화 =====
    # dual
    vis_dual = img.copy()
    yg, xg = np.where(road_gray > 0)
    vis_dual[yg, xg] = (200, 200, 200)
    yc, xc = np.where(center > 0)
    vis_dual[yc, xc] = (255, 255, 255)
    for (x1, y1, x2, y2) in blocks:
        sub = vis_dual[y1:y2, x1:x2]
        if sub.size:
            overlay = sub.copy(); overlay[:] = (0, 0, 255)
            vis_dual[y1:y2, x1:x2] = cv2.addWeighted(sub, 0.7, overlay, 0.3, 0)

    # allwhite
    vis_allwhite = np.zeros_like(img)
    vis_allwhite[yg, xg] = (255, 255, 255)

    # 경로 그리기 (선택된 병원만)
    DRAW_THICK = 3
    if best_path:
        draw_path_on_canvas(vis_dual,     center, best_path, color=(0,0,255), thick=DRAW_THICK, tol=0)
        draw_path_on_canvas(vis_allwhite, center, best_path, color=(0,0,255), thick=DRAW_THICK, tol=0)

    # 시작점/최종 목표 마커
    for canvas in (vis_dual, vis_allwhite):
        cv2.circle(canvas, S, 6, (255, 0, 0), -1)
        if best_goal:
            cv2.circle(canvas, best_goal, 6, (0, 255, 0), -1)

    title_dual = f"result_dual (center={'white' if used_white else 'skeleton'})"
    if best_label:
        title_dual += f"  [target={best_label} | cost={best_cost:.2f}]"
    else:
        title_dual += "  [NO REACHABLE TARGET]"
    cv2.imshow(title_dual, vis_dual)
    cv2.imshow("result_allwhite (pure white road, same path)", vis_allwhite)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
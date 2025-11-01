# -*- coding: utf-8 -*-
"""
중앙선(흰색) 우선 사용, 없으면 '도로 스켈레톤'을 중앙선으로 자동 대체
+ 체증 가중치 반영
+ 결과창 2개 (dual / allwhite)
+ 직선(도로 뚫는 라인) 없음: 항상 중앙선(또는 스켈레톤) 위로만 탐색

필요:
    pip install opencv-contrib-python numpy
"""

import cv2, numpy as np
from heapq import heappush, heappop

# =========================
# 이미지 경로
# =========================
img_path = "main.png"

# -------------------------
# 밝기/마스크 파라미터
# -------------------------
T_GRAY = 20              # 회색 도로: gray >= T_GRAY
STRICT_WHITE = 245        # 중앙선(엄격): RGB 모두 >= 245
CENTERLINE_DILATE = 0     # 중앙선 단절 보정 (0~1 권장)

# 스켈레톤 생성 시 도로 전처리(침식/열림) 정도
ROAD_ERODE_ITERS = 0
ROAD_OPEN_ITERS = 1

# A* 인접(대각은 끊길 때만 True로)
ALLOW_DIAGONAL = False

# -------------------------
# 체증 비용 파라미터
# -------------------------
CONGESTION_W = 12.0  # 체증 내부 배수
ENTRY_PENALTY = 30.0  # 비체증→체증 진입 고정비
INSIDE_PENALTY = 2.5   # 체증 내부 스텝당 고정비
DILATE_CONGEST = 1     # 체증 사각형 팽창(0~2)

NEI4 = [(0, -1), (0, 1), (-1, 0), (1, 0)]
NEI8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# =========================
# 유틸: 스켈레톤(1px) 만들기
# =========================
def skeletonize01(bin01):
    """입력: 0/1, 출력: 0/1 스켈레톤(가능하면 ximgproc thinning)"""
    try:
        import cv2.ximgproc as xi
        sk = xi.thinning((bin01 * 255).astype(np.uint8), xi.THINNING_ZHANGSUEN)
        return (sk > 0).astype(np.uint8)
    except Exception:
        # 간단 대체 스켈레톤
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

# =========================
# 마스크 생성 (도로/중앙선)
# =========================
def medial_axis_center(road01):
    """
    도로(0/1)에서 거리변환 능선(지역최대)을 검출해 1px 중앙선 생성.
    얇고 연속적인 '진짜 중심선'만 남기므로 원형/도넛에서 지름선(Chord)이 사라짐.
    """
    road_u8 = (road01.astype(np.uint8) * 255)
    # 바깥(벽)을 0, 도로를 255로 두고 거리변환
    dist = cv2.distanceTransform((road_u8 > 0).astype(np.uint8), cv2.DIST_L2, 3)
    # 능선: 팽창 후 자신과 같은 픽셀만 남김(지역최대)
    dist_f = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    maxmap = cv2.dilate(dist_f, kernel, 1)
    ridge = (dist_f == maxmap) & (road01 > 0)

    # 한 픽셀 얇게 정리
    ridge = ridge.astype(np.uint8)
    try:
        import cv2.ximgproc as xi
        ridge = xi.thinning(ridge * 255, xi.THINNING_ZHANGSUEN)
        ridge = (ridge > 0).astype(np.uint8)
    except Exception:
        ridge = skeletonize01(ridge)  # 대체
    return ridge


def make_road_and_center(img_bgr):
    """
    1) road_gray: 밝은 영역(회색 도로)
    2) center_white: '엄격 흰색' 중앙선 (있으면 사용)
    3) fallback_center: road_gray에서 스켈레톤(도로 중심선 대용)
    최종 center = (center_white가 충분히 크면 그걸) else (fallback_center)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    road_gray = (gray >= T_GRAY).astype(np.uint8)

    # 도로 전처리(너무 얇은 곳 넓히고 구멍 조금 메움)
    if ROAD_OPEN_ITERS > 0:
        kern = np.ones((3, 3), np.uint8)
        road_gray = cv2.morphologyEx(road_gray, cv2.MORPH_OPEN, kern, iterations=ROAD_OPEN_ITERS)
    if ROAD_ERODE_ITERS > 0:
        road_gray = cv2.erode(road_gray, np.ones((3, 3), np.uint8), iterations=ROAD_ERODE_ITERS)

    # 중앙선(엄격 흰색)
    b, g, r = cv2.split(img_bgr)
    center_white = ((b >= STRICT_WHITE) & (g >= STRICT_WHITE) & (r >= STRICT_WHITE)).astype(np.uint8)
    center_white &= road_gray  # 도로 내부만

    if CENTERLINE_DILATE > 0:
        k = 2 * CENTERLINE_DILATE + 1
        center_white = cv2.dilate(center_white, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1)

    # 스켈레톤(도로 중심선 대용)
    fallback_center = medial_axis_center(road_gray)

    # 선택 기준: 흰 중앙선 픽셀이 충분하면 그대로 사용, 아니면 스켈레톤 사용
    n_white = int(center_white.sum())
    n_fallback = int(fallback_center.sum())
    use_white = (n_white > 0)

    center = center_white if use_white else fallback_center
    return road_gray, center, center_white, fallback_center, use_white

# =========================
# 중앙선 전용 A* (체증 비용 반영)
# =========================
def astar_on_centerline(center01, start_xy, goal_xy, cong_mask=None,
                        allow_diag=False,
                        congestion_w=12.0, entry_penalty=30.0, inside_penalty=2.5):
    """
    center01: 0/1 중앙선(또는 스켈) 마스크
    start_xy, goal_xy: (x, y)
    cong_mask: 0/1 체증 마스크
    """
    H, W = center01.shape
    sx, sy = start_xy
    gx, gy = goal_xy

    ys, xs = np.where(center01 > 0)
    if len(xs) == 0:
        return []
    def snap_to_center(x, y):
        i = int(np.argmin((xs - x)**2 + (ys - y)**2))
        return int(xs[i]), int(ys[i])

    s = snap_to_center(sx, sy)
    g = snap_to_center(gx, gy)
    sx, sy = s
    gx, gy = g

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

    if (gx, gy) not in came and (gx, gy) != (sx, sy):
        return []
    
    path = []
    cur = (gx, gy)
    while cur is not None:
        path.append((cur[1], cur[0]))
        if cur == (sx, sy):
            break
        cur = came.get(cur)
    path.reverse()
    return path

# =========================
# 마우스 UI (S/D + 체증 사각형)
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
        self.blocks = []   # (x1,y1,x2,y2)

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

# ==== robust draw: 센터(또는 스켈) 위 픽셀만 최종 출력 ====
def draw_path_on_canvas(canvas, center01, path, color=(0, 0, 255), thick=3, tol=1):
    """
    canvas: BGR 이미지
    center01: 0/1 중앙선(또는 스켈) 마스크
    path: [(y,x), ...]
    tol: 센터 라인 허용 오차(px). 1~2 주면 약간 퍼진 선도 허용됨.
    """
    if len(path) < 2:
        return

    h, w = center01.shape
    # 1) 경로를 빈 마스크에 그림 (안티알리아싱 금지: LINE_8)
    path_mask = np.zeros((h, w), np.uint8)
    for (y1, x1), (y2, x2) in zip(path[:-1], path[1:]):
        cv2.line(path_mask, (x1, y1), (x2, y2), 255, thickness=thick, lineType=cv2.LINE_8)

    # 2) 센터를 약간 팽창해 허용오차 반영
    if tol > 0:
        k = 2 * tol + 1
        center_thick = cv2.dilate(center01, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1)
    else:
        center_thick = center01

    # 3) 센터 위 픽셀만 남김
    masked = cv2.bitwise_and(path_mask, (center_thick * 255).astype(np.uint8))

    # 4) 최종 합성
    ys, xs = np.where(masked > 0)
    canvas[ys, xs] = color

# =========================
# 메인
# =========================
if __name__ == "__main__":
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

    # 마스크 생성(흰 중앙선 우선, 없으면 스켈레톤)
    road_gray, center, center_white, center_fallback, used_white = make_road_and_center(img)

    # 선택 UI
    picker = Picker(img)
    cv2.namedWindow(picker.window_name)
    cv2.setMouseCallback(picker.window_name, picker.on_mouse)
    picker.redraw()

    while True:
        k = cv2.waitKey(20)
        if k in (13, 10):  # Enter
            if picker.S and picker.D:
                break
        elif k == 27:
            cv2.destroyAllWindows()
            raise SystemExit
        elif k == ord('r'):
            picker.set_mode(not picker.drawing_block_mode)
        elif k == ord('c'):
            picker.clear_blocks()

    S, D = picker.S, picker.D
    blocks = picker.blocks
    cv2.destroyWindow(picker.window_name)

    # 체증 마스크
    H, W = center.shape
    cong_mask = np.zeros((H, W), np.uint8)
    for (x1, y1, x2, y2) in blocks:
        cong_mask[y1:y2, x1:x2] = 1
    if DILATE_CONGEST > 0:
        k = 2 * DILATE_CONGEST + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        cong_mask = cv2.dilate(cong_mask, kernel, 1)

    # 중앙선/스켈레톤 위에서만 경로 탐색 (직선 없음)
    path = astar_on_centerline(
        center, S, D,
        cong_mask=cong_mask,
        allow_diag=ALLOW_DIAGONAL,
        congestion_w=CONGESTION_W,
        entry_penalty=ENTRY_PENALTY,
        inside_penalty=INSIDE_PENALTY
    )

    # ===== 시각화 =====
# ===== 시각화 =====
# (a) result_dual: 회색 도로 + 중앙선(흰) or 스켈(흰)
vis_dual = img.copy()
yg, xg = np.where(road_gray > 0)
vis_dual[yg, xg] = (200, 200, 200)  # 회색 도로
yc, xc = np.where(center > 0)
vis_dual[yc, xc] = (255, 255, 255)  # 중앙선/스켈

# 체증 사각형 반투명 오버레이 (dual에만)
for (x1, y1, x2, y2) in blocks:
    sub = vis_dual[y1:y2, x1:x2]
    if sub.size:
        overlay = sub.copy()
        overlay[:] = (0, 0, 255)
        vis_dual[y1:y2, x1:x2] = cv2.addWeighted(sub, 0.7, overlay, 0.3, 0)

# (b) result_allwhite: '원본 img'에서 새로 만들고, 도로만 순백으로 칠함
vis_allwhite = img.copy()                 # ← 핵심: dual에서 복사하지 않음
vis_allwhite[yg, xg] = (255, 255, 255)    # 도로 전체를 흰색으로

# 경로 그리기 (중앙선/스켈 경로만) — 직선 없음
DRAW_THICK = 3
TOL = 0  # 센터에서 ±0px만 허용

if len(path) >= 2:
    draw_path_on_canvas(vis_dual, center, path, color=(0, 0, 255), thick=DRAW_THICK, tol=TOL)
    draw_path_on_canvas(vis_allwhite, center, path, color=(0, 0, 255), thick=DRAW_THICK, tol=TOL)
else:
    cv2.putText(vis_dual, "No path on center/skeleton", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(vis_allwhite, "No path on center/skeleton", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# S/D 마커 (클린하게 마지막에)
for canvas in (vis_dual, vis_allwhite):
    cv2.circle(canvas, S, 6, (255, 0, 0), -1)
    cv2.circle(canvas, D, 6, (0, 255, 0), -1)

# 결과 표시
title_dual = "result_dual (gray road + {} + congestion)".format("white center" if used_white else "road skeleton")
cv2.imshow(title_dual, vis_dual)
cv2.imshow("result_allwhite (road painted white, same path)", vis_allwhite)
cv2.waitKey(0)
cv2.destroyAllWindows()

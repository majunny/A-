# -*- coding: utf-8 -*-
"""
흰색=도로(자유공간), 검은색=벽 이미지를 위한 길찾기
1) 스켈레톤 그래프에서 경로 시도 (가중 비용 반영)
2) 실패 시 픽셀 격자 A* 폴백 (가중 비용 반영)

추가 UX:
- 'R' 키: [경로 설정] / [체증 설정] 모드 전환 (한 창)
- 'C' 키: 모든 체증 영역 제거
- 체증 모드에서 좌클릭 2번으로 사각형 체증 영역 추가
- Enter: 경로 탐색
- 경로 실패 시 메시지

교통/환경 모델:
- 체증 비용 배수(CONJESTION_W) + 팽창(DILATE_RADIUS)
- 진입/내부/경계 통과 패널티(ENTRY/INSIDE/CROSS)
- 벽타기 억제:
  (a) 도로 마스크 침식(ERODE_ITERS)
  (b) 벽 근접 패널티(거리변환 기반 wall_penalty)
  (c) A* 코너-컷팅 금지(대각 이동 시 양옆 직교칸 열려야 허용)

필요:
    pip install opencv-contrib-python networkx numpy
"""

import cv2, numpy as np, networkx as nx
from math import hypot
from heapq import heappush, heappop

# =========================
# 이미지 경로
# =========================
img_path = "main.png"

# -------------------------
# 튜닝 파라미터
# -------------------------
# 체증
CONGESTION_W   = 12.0   # 체증 구역 이동 가중치 배수 (10~30)
DILATE_RADIUS  = 11     # 체증 마스크 팽창 반경 (픽셀, 7~15)
ENTRY_PENALTY  = 120.0  # 비체증→체증 진입 고정비
INSIDE_PENALTY = 8.0    # 체증 내부 이동 시 스텝당 고정비
CROSS_PENALTY  = 40.0   # 체증 경계 통과 고정비

# 벽타기 억제
ERODE_ITERS          = 0     # 도로 침식 횟수(벽에서 여유 간격 확보), 0~3
WALL_PENALTY_GAIN    = 0   # 벽 가까울수록 추가될 가중치 계수
WALL_PENALTY_DECAY   = 0   # 거리감쇠 (픽셀). 작을수록 경계 근처만 강함.

# 그래픽
DRAW_SKELETON_GRAY   = (200, 200, 200)
DRAW_ROUTE_RED       = (0, 0, 255)
DRAW_START_BLUE      = (255, 0, 0)
DRAW_GOAL_GREEN      = (0, 255, 0)

NEI8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# ------------------------------------------------
# 0) 도로(자유공간) 마스크: 흰색=1, 검은=0
# ------------------------------------------------
def road_mask_white(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Otsu로 흰/검 분리
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    road = (th == 255).astype(np.uint8)  # 흰색=1
    # 잡음 정리 및 작은 틈 메우기
    road = cv2.morphologyEx(road * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)
    road = (road > 0).astype(np.uint8)
    return road  # 0/1

# ------------------------------------------------
# 1) 스켈레톤
# ------------------------------------------------
def skeletonize01(bin01):
    try:
        import cv2.ximgproc as xi
        sk = xi.thinning((bin01 * 255).astype(np.uint8), xi.THINNING_ZHANGSUEN)
        return (sk > 0).astype(np.uint8)
    except Exception:
        # 간단 fallback
        prev = np.zeros_like(bin01)
        skel = bin01.copy()
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

def degree_map(skel01):
    K = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
    return cv2.filter2D(skel01.astype(np.uint8), -1, K)

def extract_nodes_and_map(skel01):
    deg = degree_map(skel01)
    node_mask = (skel01 > 0) & (deg != 2)
    num, lab = cv2.connectedComponents(node_mask.astype(np.uint8), 8)
    nodes, node_idmap = [], np.full(skel01.shape, -1, np.int32)
    nid = 0
    for comp in range(1, num):
        ys, xs = np.where(lab == comp)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        nodes.append((cy, cx))
        node_idmap[lab == comp] = nid
        nid += 1
    return nodes, node_idmap

def trace_edges(skel01, nodes, node_idmap,
                cost_map=None, entry_penalty=0.0, cong_mask=None,
                inside_penalty=0.0, cross_penalty=0.0):
    """
    스켈레톤 엣지(노드-노드) 추출 + 가중 길이(체증 비용/패널티) 계산
    """
    h, w = skel01.shape
    visited = np.zeros_like(skel01, np.uint8)
    edges = []

    def seg_cost(p1, p2):
        (y1, x1), (y2, x2) = p1, p2
        base = hypot(y2 - y1, x2 - x1)
        # 평균 가중치
        c = 1.0 if cost_map is None else 0.5 * (cost_map[y1, x1] + cost_map[y2, x2])
        add = 0.0
        if cong_mask is not None:
            a = cong_mask[y1, x1] > 0
            b = cong_mask[y2, x2] > 0
            if (not a) and b:
                add += entry_penalty + cross_penalty
            if a and b:
                add += inside_penalty
        return base * float(c) + add

    def poly_weighted_length(poly):
        if len(poly) < 2:
            return 0.0
        total = 0.0
        for a, b in zip(poly[:-1], poly[1:]):
            total += seg_cost(a, b)
        return total

    def walk(u_id, sy, sx):
        y, x = sy, sx
        py, px = -1, -1
        poly = []
        while True:
            if visited[y, x]:
                break
            visited[y, x] = 1
            poly.append((y, x))
            v_id = node_idmap[y, x]
            if v_id != -1 and v_id != u_id:
                full = [(nodes[u_id][0], nodes[u_id][1])] + poly + [(nodes[v_id][0], nodes[v_id][1])]
                l = poly_weighted_length(full)
                a, b = sorted([u_id, v_id])
                return (a, b, l, full)
            nbr = []
            for dy, dx in NEI8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel01[ny, nx] and (ny, nx) != (py, px):
                    nbr.append((ny, nx))
            if len(nbr) != 1:
                return None
            py, px = y, x
            y, x = nbr[0]

    for u_id, (uy, ux) in enumerate(nodes):
        ys, xs = np.where(node_idmap == u_id)
        starts = set()
        for y, x in zip(ys, xs):
            for dy, dx in NEI8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel01[ny, nx] and node_idmap[ny, nx] == -1:
                    starts.add((ny, nx))
        for sy, sx in starts:
            res = walk(u_id, sy, sx)
            if res:
                edges.append(res)

    uniq = {}
    for a, b, l, poly in edges:
        if (a, b) not in uniq or l < uniq[(a, b)][0]:
            uniq[(a, b)] = (l, poly)
    return [(a, b, l, poly) for (a, b), (l, poly) in uniq.items()]

def build_graph(nodes, edges):
    G = nx.Graph()
    for i, (y, x) in enumerate(nodes):
        G.add_node(i, pos=(x, y))
    for a, b, l, poly in edges:
        G.add_edge(a, b, weight=l, poly=poly)
    return G

def nearest_skel_pixel(pt_xy, skel01):
    x0, y0 = pt_xy
    ys, xs = np.where(skel01 > 0)
    if len(xs) == 0:
        return None
    i = int(np.argmin((xs - x0) ** 2 + (ys - y0) ** 2))
    return (int(xs[i]), int(ys[i]))  # (x,y)

def bfs_to_node(px_xy, skel01, node_idmap):
    from collections import deque
    x0, y0 = px_xy
    h, w = skel01.shape
    if not (0 <= x0 < w and 0 <= y0 < h) or skel01[y0, x0] == 0:
        return None
    if node_idmap[y0, x0] != -1:
        return int(node_idmap[y0, x0])
    seen = np.zeros_like(skel01, np.uint8)
    q = deque([(y0, x0)])
    seen[y0, x0] = 1
    while q:
        y, x = q.popleft()
        for dy, dx in NEI8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and skel01[ny, nx]:
                if node_idmap[ny, nx] != -1:
                    return int(node_idmap[ny, nx])
                seen[ny, nx] = 1
                q.append((ny, nx))
    return None

def stitch_polylines(path, G):
    coords = []
    for u, v in zip(path[:-1], path[1:]):
        poly = G.edges[u, v]['poly']
        if coords and coords[-1] == poly[0]:
            coords.extend(poly[1:])
        else:
            coords.extend(poly)
    return coords

# ------------------------------------------------
# 2) 격자 A* (가중 비용 + 코너-컷팅 금지)
# ------------------------------------------------
def astar_grid(road01, start_xy, goal_xy,
               cost_map=None, entry_penalty=0.0, cong_mask=None,
               inside_penalty=0.0, cross_penalty=0.0):
    """
    road01: 0/1 (흰색=1: 이동 가능)
    start_xy, goal_xy: (x, y)
    cost_map: 동일 크기의 float 맵 (기본=1.0, 체증/벽근접 > 1.0)
    entry_penalty / inside_penalty / cross_penalty: 체증 관련 패널티
    cong_mask: 체증(팽창 포함) 0/1 맵
    return: [(y,x), ...] 경로 픽셀 좌표
    """
    H, W = road01.shape
    sx, sy = start_xy
    gx, gy = goal_xy

    # 도로 밖이면 가장 가까운 도로 픽셀로 스냅
    def snap(x, y):
        ys, xs = np.where(road01 > 0)
        if len(xs) == 0:
            return None
        i = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
        return int(xs[i]), int(ys[i])

    if not (0 <= sx < W and 0 <= sy < H) or road01[sy, sx] == 0:
        s = snap(sx, sy)
        if s is None:
            return []
        sx, sy = s

    if not (0 <= gx < W and 0 <= gy < H) or road01[gy, gx] == 0:
        g = snap(gx, gy)
        if g is None:
            return []
        gx, gy = g

    def heur(x, y):  # admissible 유지(맨해튼)
        return abs(x - gx) + abs(y - gy)

    openh = []
    heappush(openh, (heur(sx, sy), 0.0, (sx, sy)))
    came = {}
    gscore = {(sx, sy): 0.0}

    while openh:
        f, gcost, (x, y) = heappop(openh)
        if (x, y) == (gx, gy):
            break

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if not road01[ny, nx]:
                continue

            # --- 코너-컷팅 금지: 대각 이동이면 양옆 직교칸도 열려 있어야 함 ---
            if dx and dy:
                if not (road01[y, nx] and road01[ny, x]):
                    continue
            # -----------------------------------------------------------------

            step = 1.41421356 if dx and dy else 1.0

            # 이동 비용 = 평균 cost * step + 패널티
            if cost_map is None:
                w = 1.0
            else:
                w = 0.5 * (float(cost_map[y, x]) + float(cost_map[ny, nx]))

            add = 0.0
            if cong_mask is not None:
                a = cong_mask[y, x]  > 0
                b = cong_mask[ny, nx] > 0
                if (not a) and b:
                    add += entry_penalty + cross_penalty
                if a and b:
                    add += inside_penalty

            ng = gcost + step * w + add

            if ng < gscore.get((nx, ny), float("inf")):
                gscore[(nx, ny)] = ng
                came[(nx, ny)] = (x, y)
                heappush(openh, (ng + heur(nx, ny), ng, (nx, ny)))

    # 경로 복원
    if (gx, gy) not in came and (gx, gy) != (sx, sy):
        return []
    path = [(sy, sx)]
    cur = (gx, gy)
    while cur != (sx, sy):
        path.append((cur[1], cur[0]))  # (y,x)
        prev = came.get(cur)
        if prev is None:
            break
        cur = prev
    path.reverse()
    return path

# ------------------------------------------------
# 3) 마우스로 S/D/체증 찍기
# ------------------------------------------------
class Picker:
    def __init__(self, img):
        self.img0 = img.copy()
        self.img = img.copy()
        self.S = None
        self.D = None

        self.window_name = "Pathfinder" # 고정 창 이름
        self.drawing_block_mode = False  # 'R' 키로 토글
        self.block_p1 = None             # 체증 영역 첫 클릭 지점
        self.blocks = []                 # (x1, y1, x2, y2) 리스트

    def on_mouse(self, e, x, y, flags, param):
        if self.drawing_block_mode:
            # 체증(장애물) 설정 모드
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
            # S/D 설정 모드
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

        # S/D 그리기
        if self.S is not None:
            cv2.circle(self.img, self.S, 6, DRAW_START_BLUE, -1)
        if self.D is not None:
            cv2.circle(self.img, self.D, 6, DRAW_GOAL_GREEN, -1)

        # 체증 사각형
        for (x1, y1, x2, y2) in self.blocks:
            cv2.rectangle(self.img, (x1, y1), (x2, y2), DRAW_ROUTE_RED, 2)

        # 체증 첫 점
        if self.block_p1 is not None:
            cv2.circle(self.img, self.block_p1, 5, DRAW_ROUTE_RED, -1)

        # 모드 텍스트
        if self.drawing_block_mode:
            mode_text = "Mode: BLOCK (R: Switch, C: Clear)"
            color = (0, 255, 255)
        else:
            mode_text = "Mode: S/D (R: Switch)"
            color = (255, 255, 0)

        cv2.putText(self.img, mode_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow(self.window_name, self.img)

# ------------------------------------------------
# 4) 메인 solve (체증 + 벽타기 억제)
# ------------------------------------------------
def solve(image_bgr, S_xy, D_xy, blocks=[], show_no_path_marker=True):
    # (A) 흰색=도로
    road = road_mask_white(image_bgr).astype(np.uint8)  # 0/1
    H, W = road.shape

    # (A.1) 벽타기 억제 (a): 도로 침식으로 벽에서 여유 간격 확보
    if ERODE_ITERS > 0:
        road = cv2.erode(road, np.ones((3, 3), np.uint8), iterations=ERODE_ITERS)

    # (A.2) 체증 마스크(원본) + 팽창
    cong_mask = np.zeros((H, W), dtype=np.uint8)
    for (x1, y1, x2, y2) in blocks:
        cong_mask[y1:y2, x1:x2] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*DILATE_RADIUS+1, 2*DILATE_RADIUS+1))
    cong_mask_dil = cv2.dilate(cong_mask, kernel, iterations=1)

    # (A.3) 비용맵 기본 1.0
    cost_map = np.ones((H, W), dtype=np.float32)
    # 체증 내부 가중치
    cost_map[cong_mask_dil > 0] *= CONGESTION_W

    # (A.4) 벽타기 억제 (b): 벽 근접 패널티 (distance transform)
    # road==1 내부 거리맵: 값이 클수록 벽에서 멀리 있음
    dist = cv2.distanceTransform((road*255).astype(np.uint8), cv2.DIST_L2, 3)
    # 0에 가까울수록 벽에 붙음 → exp(-dist/decay)로 패널티 형성
    wall_penalty = np.exp(-dist / max(1e-6, WALL_PENALTY_DECAY))
    cost_map += wall_penalty * WALL_PENALTY_GAIN

    # (B) 스켈레톤 경로 (가중 비용)
    skel = skeletonize01(road)
    nodes, node_idmap = extract_nodes_and_map(skel)
    edges = trace_edges(
        skel, nodes, node_idmap,
        cost_map=cost_map,
        entry_penalty=ENTRY_PENALTY,
        cong_mask=cong_mask_dil,
        inside_penalty=INSIDE_PENALTY,
        cross_penalty=CROSS_PENALTY
    )
    G = build_graph(nodes, edges)

    vis = image_bgr.copy()
    # 스켈 회색 표시
    ys, xs = np.where(skel > 0)
    vis[ys, xs] = DRAW_SKELETON_GRAY

    # 체증 영역 반투명 표시(원 사각형 기준)
    for (x1, y1, x2, y2) in blocks:
        sub = vis[y1:y2, x1:x2]
        red_rect = np.zeros_like(sub)
        red_rect[:,:] = DRAW_ROUTE_RED
        res = cv2.addWeighted(sub, 0.7, red_rect, 0.3, 0.0)
        vis[y1:y2, x1:x2] = res
        cv2.rectangle(vis, (x1, y1), (x2, y2), DRAW_ROUTE_RED, 1)

    red_path_drawn = False

    if len(G) > 0 and len(nodes) >= 2:
        S_px = nearest_skel_pixel(S_xy, skel)
        D_px = nearest_skel_pixel(D_xy, skel)
        if S_px and D_px:
            s_idx = bfs_to_node(S_px, skel, node_idmap)
            d_idx = bfs_to_node(D_px, skel, node_idmap)
            if s_idx is not None and d_idx is not None:
                try:
                    _, node_path = nx.bidirectional_dijkstra(G, s_idx, d_idx, weight='weight')
                    route = stitch_polylines(node_path, G)
                    if len(route) >= 2:
                        for (y1, x1), (y2, x2) in zip(route[:-1], route[1:]):
                            cv2.line(vis, (x1, y1), (x2, y2), DRAW_ROUTE_RED, 3)
                        red_path_drawn = True
                except Exception:
                    pass

    # (C) 스켈 실패 → A* 폴백 (가중 비용 + 코너-컷팅 금지)
    if not red_path_drawn:
        grid_path = astar_grid(
            road, S_xy, D_xy,
            cost_map=cost_map,
            entry_penalty=ENTRY_PENALTY,
            cong_mask=cong_mask_dil,
            inside_penalty=INSIDE_PENALTY,
            cross_penalty=CROSS_PENALTY
        )
        if len(grid_path) >= 2:
            for (y1, x1), (y2, x2) in zip(grid_path[:-1], grid_path[1:]):
                cv2.line(vis, (x1, y1), (x2, y2), DRAW_ROUTE_RED, 3)
            red_path_drawn = True

    # (D) 실패 메시지
    if not red_path_drawn:
        cv2.putText(vis, "No detour exists", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, DRAW_ROUTE_RED, 2, cv2.LINE_AA)

    # S/D 마커
    cv2.circle(vis, S_xy, 6, DRAW_START_BLUE, -1)
    cv2.circle(vis, D_xy, 6, DRAW_GOAL_GREEN, -1)
    return vis, (skel * 255).astype(np.uint8), (road * 255).astype(np.uint8)

# ------------------------------------------------
# 5) 실행
# ------------------------------------------------
if __name__ == "__main__":
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

    picker = Picker(img)
    cv2.namedWindow(picker.window_name)
    cv2.setMouseCallback(picker.window_name, picker.on_mouse)
    picker.redraw()

    while True:
        k = cv2.waitKey(20)
        if k in (13, 10):  # Enter
            if picker.S and picker.D:
                break
        elif k == 27:  # Esc
            cv2.destroyAllWindows()
            raise SystemExit
        elif k == ord('r'):  # 모드 전환
            picker.set_mode(not picker.drawing_block_mode)
        elif k == ord('c'):  # 체증 초기화
            picker.clear_blocks()

    S, D = picker.S, picker.D
    blocks = picker.blocks
    cv2.destroyWindow(picker.window_name)

    vis, skel_img, road_mask = solve(img, S, D, blocks)
    cv2.imshow("road_mask (eroded)", road_mask)
    cv2.imshow("skeleton", skel_img)
    cv2.imshow("result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

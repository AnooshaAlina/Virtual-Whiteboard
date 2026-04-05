import cv2
import numpy as np
import mediapipe as mp
import math

# ── Helpers ──────────────────────────────────────────────────────────────────

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def rounded_rect(img, x1, y1, x2, y2, r, color, filled=True):
    t = -1 if filled else 2
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, t)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, t)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(img, (cx, cy), r, color, t)

# ── Constants ────────────────────────────────────────────────────────────────

W, H      = 1280, 720
TOOLBAR_H = 70
CANVAS_H  = H - TOOLBAR_H

PALETTE_COLORS = [
    (255,255,255),(30,30,30),(50,50,220),(30,140,255),
    (30,220,255),(50,200,80),(220,210,30),(220,80,50),
    (180,50,160),(180,100,255),
]
PALETTE_NAMES = ["White","Black","Red","Orange","Yellow","Green","Cyan","Blue","Purple","Pink"]

BRUSH_SIZES   = [3, 6, 10, 16, 24]
ERASER_R      = 40

TOOLBAR_BG    = (20, 20, 20)
ACCENT        = (0, 200, 255)

COLOR_R       = 16
COLOR_GAP     = 8
COLOR_ROW_Y   = TOOLBAR_H // 2
COLOR_START_X = 160

BRUSH_W       = 34
BRUSH_H       = 34
BRUSH_START_X = COLOR_START_X + len(PALETTE_COLORS)*(COLOR_R*2+COLOR_GAP) + 20
BRUSH_ROW_Y   = TOOLBAR_H//2 - BRUSH_H//2

CLEAR_BTN = (W-200, 15, W-120, 55)
UNDO_BTN  = (W-110, 15, W-20,  55)

# ── State ────────────────────────────────────────────────────────────────────

colorIndex   = 1
brushIndex   = 1
MODE         = "IDLE"
prev_cursor  = None
shape_start  = None

strokes      = []   # committed strokes
live_pts     = []   # points of the stroke being drawn right now
live_color   = None
live_size    = None

paintCanvas  = np.ones((CANVAS_H, W, 3), dtype=np.uint8) * 255

# ── Mediapipe ────────────────────────────────────────────────────────────────

mpHands = mp.solutions.hands
hands   = mpHands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)
mpDraw  = mp.solutions.drawing_utils
HS = mpDraw.DrawingSpec(color=(0,220,255), thickness=2, circle_radius=3)
CS = mpDraw.DrawingSpec(color=(200,200,200), thickness=1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

# ── Toolbar ──────────────────────────────────────────────────────────────────

def draw_toolbar(img):
    cv2.rectangle(img, (0,0), (W, TOOLBAR_H), TOOLBAR_BG, -1)
    cv2.line(img, (0, TOOLBAR_H), (W, TOOLBAR_H), (60,60,60), 1)

    mc = {"DRAW":(0,200,100),"ERASE":(0,80,255),"SHAPE":ACCENT,
          "IDLE":(120,120,120),"SELECT":(255,180,0)}.get(MODE,(200,200,200))
    cv2.putText(img, MODE, (12,26), cv2.FONT_HERSHEY_DUPLEX, 0.65, mc, 1, cv2.LINE_AA)
    hint = {"DRAW":"index finger","ERASE":"open palm","SHAPE":"index+middle",
            "SELECT":"pinch","IDLE":""}.get(MODE,"")
    cv2.putText(img, hint, (12,50), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (140,140,140), 1, cv2.LINE_AA)

    for i, bgr in enumerate(PALETTE_COLORS):
        cx = COLOR_START_X + i*(COLOR_R*2+COLOR_GAP) + COLOR_R
        if i == colorIndex:
            cv2.circle(img, (cx, COLOR_ROW_Y), COLOR_R+4, ACCENT, 2)
        cv2.circle(img, (cx, COLOR_ROW_Y), COLOR_R, bgr, -1)
        cv2.circle(img, (cx, COLOR_ROW_Y), COLOR_R, (80,80,80), 1)

    for i, sz in enumerate(BRUSH_SIZES):
        bx = BRUSH_START_X + i*(BRUSH_W+6)
        col = ACCENT if i == brushIndex else (60,60,60)
        rounded_rect(img, bx, BRUSH_ROW_Y, bx+BRUSH_W, BRUSH_ROW_Y+BRUSH_H, 6, col)
        r = max(2, min(sz//2, 11))
        dot_col = PALETTE_COLORS[colorIndex] if i != brushIndex else (20,20,20)
        cv2.circle(img, (bx+BRUSH_W//2, BRUSH_ROW_Y+BRUSH_H//2), r, dot_col, -1)

    rounded_rect(img, *CLEAR_BTN, 8, (60,30,30))
    cv2.putText(img,"Clear",(CLEAR_BTN[0]+14,CLEAR_BTN[1]+27),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,100,100),1,cv2.LINE_AA)
    rounded_rect(img, *UNDO_BTN, 8, (30,50,80))
    cv2.putText(img,"Undo",(UNDO_BTN[0]+16,UNDO_BTN[1]+27),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(100,170,230),1,cv2.LINE_AA)

def toolbar_hit(cursor):
    global colorIndex, brushIndex
    cx, cy = cursor
    for i in range(len(PALETTE_COLORS)):
        sx = COLOR_START_X + i*(COLOR_R*2+COLOR_GAP) + COLOR_R
        if dist(cursor,(sx, COLOR_ROW_Y)) < COLOR_R+5:
            colorIndex = i; return "COLOR"
    for i in range(len(BRUSH_SIZES)):
        bx = BRUSH_START_X + i*(BRUSH_W+6)
        if bx < cx < bx+BRUSH_W and BRUSH_ROW_Y < cy < BRUSH_ROW_Y+BRUSH_H:
            brushIndex = i; return "BRUSH"
    if CLEAR_BTN[0]<cx<CLEAR_BTN[2] and CLEAR_BTN[1]<cy<CLEAR_BTN[3]: return "CLEAR"
    if UNDO_BTN[0]<cx<UNDO_BTN[2]   and UNDO_BTN[1]<cy<UNDO_BTN[3]:   return "UNDO"
    return None

# ── Canvas helpers ───────────────────────────────────────────────────────────

def redraw_canvas():
    global paintCanvas
    paintCanvas[:] = 255
    for s in strokes:
        _apply_stroke(s)

def _apply_stroke(s):
    if s["type"] == "ERASE":
        for pt in s["pts"]:
            cv2.circle(paintCanvas, pt, ERASER_R, (255,255,255), -1)
    elif s["type"] == "DRAW":
        pts = s["pts"]
        for i in range(1, len(pts)):
            cv2.line(paintCanvas, pts[i-1], pts[i],
                     s["color"], s["size"], cv2.LINE_AA)
    elif s["type"] == "RECT":
        cv2.rectangle(paintCanvas, s["pts"][0], s["pts"][1],
                      s["color"], s["size"], cv2.LINE_AA)

def commit_live():
    global live_pts, live_color, live_size
    if live_pts and len(live_pts) > 1:
        strokes.append({"type":"DRAW","color":live_color,
                        "size":live_size,"pts":list(live_pts)})
    live_pts  = []
    live_color = None
    live_size  = None

# ── Main loop ────────────────────────────────────────────────────────────────

print("Virtual Whiteboard — Q=quit | Z=undo | C=clear")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame  = cv2.flip(frame, 1)
    frame  = cv2.resize(frame, (W, H))
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # ── Composite: strokes over frosted camera ────────────────────────────
    display   = frame.copy()
    cam_roi   = frame[TOOLBAR_H:, :]

    # Slight white wash on camera so strokes pop
    frosted = cv2.addWeighted(cam_roi, 0.7, np.ones_like(cam_roi)*255, 0.3, 0)

    # Paste paint strokes at full opacity where canvas is not white
    gray_p = cv2.cvtColor(paintCanvas, cv2.COLOR_BGR2GRAY)
    _, bg  = cv2.threshold(gray_p, 240, 255, cv2.THRESH_BINARY)
    fg     = cv2.bitwise_not(bg)

    blended = frosted.copy()
    blended[fg > 0] = paintCanvas[fg > 0]

    display[TOOLBAR_H:, :] = blended
    draw_toolbar(display)

    # ── Hand ─────────────────────────────────────────────────────────────
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm   = [(int(p.x*W), int(p.y*H)) for p in hand.landmark]
        mpDraw.draw_landmarks(display, hand, mpHands.HAND_CONNECTIONS, HS, CS)

        index_tip = lm[8]
        thumb_tip = lm[4]
        wrist     = lm[0]

        iu  = lm[8][1]  < lm[6][1]    # index up
        mu  = lm[12][1] < lm[10][1]   # middle up
        ru  = lm[16][1] < lm[14][1]   # ring up
        pu  = lm[20][1] < lm[18][1]   # pinky up

        # Open palm: all 4 fingers extended AND their tips are farther
        # from wrist than their PIP knuckles (robust to camera angle)
        tip_ids = [8,  12, 16, 20]
        pip_ids = [6,  10, 14, 18]
        open_palm = (iu and mu and ru and pu and
                     all(dist(lm[t], wrist) > dist(lm[p], wrist)
                         for t, p in zip(tip_ids, pip_ids)))

        pinch = dist(index_tip, thumb_tip) < 38

        # ── Smooth cursor ─────────────────────────────────────────────────
        if prev_cursor is None:
            prev_cursor = index_tip
        a = 0.45
        cursor = (int(prev_cursor[0]*(1-a) + index_tip[0]*a),
                  int(prev_cursor[1]*(1-a) + index_tip[1]*a))
        prev_cursor = cursor

        # ── Mode ──────────────────────────────────────────────────────────
        old_mode = MODE
        if open_palm:
            MODE = "ERASE"
        elif pinch:
            MODE = "SELECT"
        elif iu and mu and not ru:
            MODE = "SHAPE"
        elif iu and not mu:
            MODE = "DRAW"
        else:
            MODE = "IDLE"

        # Commit live stroke when leaving DRAW
        if old_mode == "DRAW" and MODE != "DRAW":
            commit_live()

        # ── Toolbar zone ──────────────────────────────────────────────────
        if cursor[1] < TOOLBAR_H:
            action = toolbar_hit(cursor)
            if action == "CLEAR":
                strokes.clear()
                paintCanvas[:] = 255
                live_pts = []
            elif action == "UNDO":
                if strokes:
                    strokes.pop()
                    redraw_canvas()

        else:
            # canvas-space point (no toolbar offset subtraction bug)
            cc = (cursor[0], cursor[1] - TOOLBAR_H)

            # ── DRAW: incremental — only add last segment ──────────────────
            if MODE == "DRAW":
                if not live_pts:
                    live_color = PALETTE_COLORS[colorIndex]
                    live_size  = BRUSH_SIZES[brushIndex]
                live_pts.append(cc)
                if len(live_pts) >= 2:
                    cv2.line(paintCanvas, live_pts[-2], live_pts[-1],
                             live_color, live_size, cv2.LINE_AA)

            # ── ERASE ─────────────────────────────────────────────────────
            elif MODE == "ERASE":
                # Draw white circle directly on the canvas
                cv2.circle(paintCanvas, cc, ERASER_R, (255,255,255), -1)
                strokes.append({"type":"ERASE","pts":[cc],
                                "color":None,"size":ERASER_R})
                # Show eraser circle on display
                cv2.circle(display, cursor, ERASER_R, (0,80,255), 2, cv2.LINE_AA)
                cv2.circle(display, cursor, ERASER_R, (0,80,255,80), -1)

            # ── SHAPE: live rect preview on display only ───────────────────
            elif MODE == "SHAPE":
                if old_mode != "SHAPE":
                    shape_start = cc
                if shape_start:
                    ds = (shape_start[0], shape_start[1] + TOOLBAR_H)
                    de = (cc[0],          cc[1]          + TOOLBAR_H)
                    cv2.rectangle(display, ds, de,
                                  PALETTE_COLORS[colorIndex],
                                  BRUSH_SIZES[brushIndex], cv2.LINE_AA)

            # ── Commit rect when leaving SHAPE ────────────────────────────
            if old_mode == "SHAPE" and MODE != "SHAPE" and shape_start:
                cv2.rectangle(paintCanvas, shape_start, cc,
                              PALETTE_COLORS[colorIndex],
                              BRUSH_SIZES[brushIndex], cv2.LINE_AA)
                strokes.append({"type":"RECT",
                                "color":PALETTE_COLORS[colorIndex],
                                "size":BRUSH_SIZES[brushIndex],
                                "pts":[shape_start, cc]})
                shape_start = None

            if MODE not in ("SHAPE",):
                if old_mode != "SHAPE":
                    shape_start = None

        # ── Cursor indicator ──────────────────────────────────────────────
        if MODE != "ERASE":
            ring_col = {"DRAW":PALETTE_COLORS[colorIndex],
                        "SHAPE":ACCENT,"SELECT":(255,180,0)}.get(MODE,(180,180,180))
            cv2.circle(display, cursor, 12, ring_col, 2, cv2.LINE_AA)
            cv2.circle(display, cursor, 3,  ring_col, -1, cv2.LINE_AA)

    else:
        commit_live()
        prev_cursor = None

    cv2.imshow("Virtual Whiteboard", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('z'):
        if strokes: strokes.pop(); redraw_canvas()
    elif key == ord('c'):
        strokes.clear(); paintCanvas[:] = 255; live_pts = []

cap.release()
cv2.destroyAllWindows()

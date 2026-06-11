# app.py
# 실행:
# pip install streamlit pandas numpy plotly
# streamlit run app.py

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="AI 융합 지진 진원 찾기", layout="wide")

P_SPEED = 6.0   # km/s, 단순화한 P파 속도
S_SPEED = 3.5   # km/s, 단순화한 S파 속도

st.title("🌍 지구과학 × 수학 × AI 융합: 지진 진원 찾기")
st.caption("세 관측소의 좌표와 PS시를 이용해 진원거리 원을 그리고, 교점과 공통현으로 진원을 추정합니다.")

with st.expander("수업 활동 안내", expanded=True):
    st.markdown("""
1. 실제 지진 데이터를 보고 진앙의 위도·경도를 확인합니다.
2. 학생들이 세 관측소의 위도·경도를 조사해 입력합니다.
3. 각 관측소의 PS시, 즉 S파 도착 시각과 P파 도착 시각의 차이를 입력합니다.
4. PS시로 진원거리를 계산합니다.
5. 각 관측소를 중심, 진원거리를 반지름으로 하는 세 원을 좌표평면에 그립니다.
6. 세 원의 교점 또는 공통현이 만나는 점을 이용해 진원을 추정합니다.
""")

st.sidebar.header("⚙️ 기본 설정")
p_speed = st.sidebar.number_input("P파 속도(km/s)", value=P_SPEED, min_value=1.0, step=0.1)
s_speed = st.sidebar.number_input("S파 속도(km/s)", value=S_SPEED, min_value=1.0, step=0.1)

st.sidebar.markdown("PS시 → 진원거리 공식")
st.sidebar.latex(r"d=\frac{V_PV_S}{V_P-V_S}\times (T_S-T_P)")

def ps_to_distance(ps_seconds, vp, vs):
    return (vp * vs / (vp - vs)) * ps_seconds

def latlon_to_xy_km(lat, lon, lat0, lon0):
    x = (lon - lon0) * 111.32 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110.57
    return x, y

def xy_km_to_latlon(x, y, lat0, lon0):
    lat = y / 110.57 + lat0
    lon = x / (111.32 * math.cos(math.radians(lat0))) + lon0
    return lat, lon

def circle_intersections(c1, r1, c2, r2):
    x0, y0 = c1
    x1, y1 = c2
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)

    if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
        return []

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h2 = r1**2 - a**2
    if h2 < 0:
        h2 = 0

    h = math.sqrt(h2)
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d

    rx = -dy * h / d
    ry = dx * h / d

    return [(xm + rx, ym + ry), (xm - rx, ym - ry)]

def estimate_epicenter_least_squares(points, radii):
    """
    원 방정식:
    (x-xi)^2 + (y-yi)^2 = ri^2

    첫 번째 원을 기준으로 빼면 선형식이 됨.
    Ax = b 형태로 최소제곱 추정.
    """
    x1, y1 = points[0]
    r1 = radii[0]

    A = []
    b = []
    for i in range(1, len(points)):
        xi, yi = points[i]
        ri = radii[i]

        A.append([2 * (xi - x1), 2 * (yi - y1)])
        b.append((x1**2 + y1**2 - r1**2) - (xi**2 + yi**2 - ri**2))

    A = np.array(A)
    b = np.array(b)

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[0], sol[1]

st.subheader("1️⃣ 실제 지진 데이터")

sample_quakes = pd.DataFrame({
    "발생시각": [
        "2026/06/10 03:16:57",
        "2026/06/08 09:07:14",
        "2026/04/05 22:10:26",
        "2026/03/14 09:00:20",
    ],
    "규모": [2.9, 2.9, 2.4, 2.6],
    "깊이(km)": [20, 5, 11, 11],
    "위도": [35.67, 34.57, 35.08, 35.82],
    "경도": [129.60, 128.28, 127.07, 128.35],
    "위치": [
        "울산 북구 동북동쪽 24km 해역",
        "경남 통영시 남남서쪽 35km 해역",
        "전남 화순군 동북동쪽 8km 지역",
        "경북 성주군 남남동쪽 13km 지역",
    ]
})

uploaded = st.file_uploader(
    "기상청 자료를 CSV로 내려받아 업로드할 수 있습니다. 없으면 예시 데이터를 사용합니다.",
    type=["csv"]
)

if uploaded:
    quake_df = pd.read_csv(uploaded)
else:
    quake_df = sample_quakes

st.dataframe(quake_df, use_container_width=True)

selected_idx = st.selectbox(
    "탐구할 지진 선택",
    quake_df.index,
    format_func=lambda i: f"{quake_df.loc[i, '발생시각']} | M{quake_df.loc[i, '규모']} | {quake_df.loc[i, '위치']}"
)

quake = quake_df.loc[selected_idx]
true_lat = float(quake["위도"])
true_lon = float(quake["경도"])

st.info(f"선택한 실제 진앙: 위도 {true_lat}, 경도 {true_lon} / {quake['위치']}")

st.subheader("2️⃣ 세 관측소 입력")

default_stations = pd.DataFrame({
    "관측소": ["서울", "대전", "부산"],
    "위도": [37.5665, 36.3504, 35.1796],
    "경도": [126.9780, 127.3845, 129.0756],
    "PS시_초": [42.0, 30.0, 18.0],
})

stations = st.data_editor(
    default_stations,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "관측소": st.column_config.TextColumn("관측소"),
        "위도": st.column_config.NumberColumn("위도", format="%.5f"),
        "경도": st.column_config.NumberColumn("경도", format="%.5f"),
        "PS시_초": st.column_config.NumberColumn("PS시(초)", min_value=0.0, step=0.1),
    }
)

stations["진원거리_km"] = stations["PS시_초"].apply(lambda t: ps_to_distance(t, p_speed, s_speed))

st.subheader("3️⃣ PS시로 진원거리 계산")
st.dataframe(stations, use_container_width=True)

lat0 = stations["위도"].mean()
lon0 = stations["경도"].mean()

xy_points = []
for _, row in stations.iterrows():
    x, y = latlon_to_xy_km(row["위도"], row["경도"], lat0, lon0)
    xy_points.append((x, y))

radii = stations["진원거리_km"].to_numpy()

try:
    est_x, est_y = estimate_epicenter_least_squares(xy_points, radii)
    est_lat, est_lon = xy_km_to_latlon(est_x, est_y, lat0, lon0)
except Exception:
    est_x, est_y, est_lat, est_lon = None, None, None, None

true_x, true_y = latlon_to_xy_km(true_lat, true_lon, lat0, lon0)

st.subheader("4️⃣ 좌표평면에 세 원 그리기")

fig = go.Figure()

theta = np.linspace(0, 2 * np.pi, 400)

for i, row in stations.iterrows():
    cx, cy = xy_points[i]
    r = radii[i]

    fig.add_trace(go.Scatter(
        x=cx + r * np.cos(theta),
        y=cy + r * np.sin(theta),
        mode="lines",
        name=f"{row['관측소']} 원"
    ))

    fig.add_trace(go.Scatter(
        x=[cx],
        y=[cy],
        mode="markers+text",
        text=[row["관측소"]],
        textposition="top center",
        marker=dict(size=10),
        name=f"{row['관측소']} 관측소"
    ))

all_intersections = []
pairs = [(0, 1), (1, 2), (0, 2)]

for a, b in pairs:
    inters = circle_intersections(xy_points[a], radii[a], xy_points[b], radii[b])
    for p in inters:
        all_intersections.append(p)
    if len(inters) == 2:
        fig.add_trace(go.Scatter(
            x=[inters[0][0], inters[1][0]],
            y=[inters[0][1], inters[1][1]],
            mode="lines",
            line=dict(dash="dash"),
            name=f"{stations.loc[a, '관측소']}-{stations.loc[b, '관측소']} 공통현"
        ))

if all_intersections:
    ix = [p[0] for p in all_intersections]
    iy = [p[1] for p in all_intersections]
    fig.add_trace(go.Scatter(
        x=ix,
        y=iy,
        mode="markers",
        marker=dict(size=8, symbol="x"),
        name="원들의 교점"
    ))

if est_x is not None:
    fig.add_trace(go.Scatter(
        x=[est_x],
        y=[est_y],
        mode="markers+text",
        text=["추정 진원"],
        textposition="bottom center",
        marker=dict(size=14, symbol="star"),
        name="추정 진원"
    ))

fig.add_trace(go.Scatter(
    x=[true_x],
    y=[true_y],
    mode="markers+text",
    text=["실제 진앙"],
    textposition="top center",
    marker=dict(size=14, symbol="diamond"),
    name="실제 진앙"
))

fig.update_layout(
    width=900,
    height=700,
    xaxis_title="동서 방향 거리 x(km)",
    yaxis_title="남북 방향 거리 y(km)",
    yaxis_scaleanchor="x",
    template="plotly_white",
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("5️⃣ 결과 해석")

if est_x is not None:
    error = math.hypot(est_x - true_x, est_y - true_y)

    col1, col2, col3 = st.columns(3)
    col1.metric("추정 진원 위도", f"{est_lat:.5f}")
    col2.metric("추정 진원 경도", f"{est_lon:.5f}")
    col3.metric("실제 진앙과 오차", f"{error:.2f} km")

    st.markdown(f"""
### AI 탐구 질문 예시
- PS시가 1초만 달라져도 진원 위치는 얼마나 변할까?
- P파와 S파 속도 값을 다르게 설정하면 결과가 어떻게 달라질까?
- 세 원이 한 점에서 만나지 않는 이유는 무엇일까?
- 관측소를 바꾸면 오차가 줄어들까?
- AI에게 오차 원인을 설명하게 한 뒤, 학생 설명과 비교해 보자.
""")
else:
    st.warning("진원을 계산할 수 없습니다. 관측소 좌표나 PS시 값을 확인하세요.")

st.subheader("6️⃣ 학생용 보고서 문장 자동 생성")

if est_x is not None:
    report = f"""
우리는 선택한 지진 자료를 바탕으로 세 관측소의 위도와 경도를 좌표평면의 좌표로 변환하였다.
각 관측소에서 구한 PS시를 이용하여 진원거리를 계산하였고,
이를 반지름으로 하는 세 원을 그렸다.

세 원은 측정 오차와 단순화된 파동 속도 때문에 정확히 한 점에서 만나지 않을 수 있었다.
따라서 원들의 교점과 공통현을 이용하여 진원의 위치를 추정하였다.

이번 탐구에서 추정한 진원은 위도 {est_lat:.5f}, 경도 {est_lon:.5f}이고,
기상청 자료의 실제 진앙과의 오차는 약 {error:.2f} km였다.
"""
    st.text_area("보고서 초안", report, height=220)

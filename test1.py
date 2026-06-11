import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="지진 진원 찾기", layout="wide")

st.title("🌏 지진 진원 찾기: 지구과학 × 수학 융합수업")

# -----------------------------
# 1. 수업용 데이터
# x = 위도, y = 경도
# -----------------------------
stations = {
    "서울 관측소": {"lat": 37.57, "lon": 126.98},
    "부산 관측소": {"lat": 35.18, "lon": 129.08},
    "광주 관측소": {"lat": 35.16, "lon": 126.85},
}

actual_epicenter = {
    "lat": 36.35,
    "lon": 127.38,
}

KM_PER_DEGREE = 111  # 수업용 근사값

for name, s in stations.items():
    dx = s["lat"] - actual_epicenter["lat"]
    dy = s["lon"] - actual_epicenter["lon"]
    degree_distance = np.sqrt(dx**2 + dy**2)
    s["distance_km"] = round(degree_distance * KM_PER_DEGREE, 1)
    s["radius_degree"] = degree_distance
    s["ps_time"] = round(s["distance_km"] / 8, 1)  # 예시용: PS시와 진원거리 관계 단순화


# -----------------------------
# 2. 세션 상태
# -----------------------------
if "student_points" not in st.session_state:
    st.session_state.student_points = {}

if "circles" not in st.session_state:
    st.session_state.circles = {}

if "show_answer" not in st.session_state:
    st.session_state.show_answer = False


# -----------------------------
# 3. 수학 함수
# -----------------------------
def parse_circle_equation(eq):
    """
    입력 예:
    (x-37.57)^2 + (y-126.98)^2 = 1.32^2
    (x-37.57)^2+(y-126.98)^2=1.74
    """
    eq = eq.replace(" ", "")

    pattern = r"\(x([+-]\d+\.?\d*)\)\^2\+\(y([+-]\d+\.?\d*)\)\^2=(\d+\.?\d*)(\^2)?"
    match = re.match(pattern, eq)

    if not match:
        return None

    x_part = float(match.group(1))
    y_part = float(match.group(2))
    value = float(match.group(3))
    squared = match.group(4)

    h = -x_part
    k = -y_part
    r = value if squared else np.sqrt(value)

    return h, k, r


def circle_points(h, k, r, n=300):
    t = np.linspace(0, 2 * np.pi, n)
    x = h + r * np.cos(t)
    y = k + r * np.sin(t)
    return x, y


def circle_intersections(c1, c2):
    x0, y0, r0 = c1
    x1, y1, r1 = c2

    dx = x1 - x0
    dy = y1 - y0
    d = np.sqrt(dx**2 + dy**2)

    if d == 0 or d > r0 + r1 or d < abs(r0 - r1):
        return []

    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h_sq = r0**2 - a**2

    if h_sq < 0:
        return []

    h = np.sqrt(h_sq)
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d

    xs1 = xm + h * dy / d
    ys1 = ym - h * dx / d
    xs2 = xm - h * dy / d
    ys2 = ym + h * dx / d

    return [(xs1, ys1), (xs2, ys2)]


def radical_axis(c1, c2):
    x1, y1, r1 = c1
    x2, y2, r2 = c2

    # Ax + By + C = 0
    A = 2 * (x2 - x1)
    B = 2 * (y2 - y1)
    C = x1**2 + y1**2 - r1**2 - x2**2 - y2**2 + r2**2

    return A, B, C


def line_intersection(l1, l2):
    A1, B1, C1 = l1
    A2, B2, C2 = l2

    det = A1 * B2 - A2 * B1

    if abs(det) < 1e-9:
        return None

    x = (B1 * C2 - B2 * C1) / det
    y = (C1 * A2 - C2 * A1) / det

    return x, y


def add_base_layout(fig):
    fig.update_layout(
        height=650,
        xaxis_title="위도 x",
        yaxis_title="경도 y",
        xaxis=dict(range=[34.5, 38.2], dtick=0.5),
        yaxis=dict(range=[126.3, 129.6], dtick=0.5),
        dragmode="pan",
        hovermode="closest",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# -----------------------------
# 4. 활동 안내
# -----------------------------
with st.expander("📌 관측소 정보 보기", expanded=True):
    info_df = pd.DataFrame([
        {
            "관측소": name,
            "위도 x": s["lat"],
            "경도 y": s["lon"],
        }
        for name, s in stations.items()
    ])
    st.dataframe(info_df, use_container_width=True)

st.divider()

# -----------------------------
# 5. 좌표평면에 관측소 위치 찍기
# -----------------------------
st.subheader("1단계. 관측소 위치를 좌표평면에 찍기")

st.caption("아래 좌표 입력칸에 관측소의 위도와 경도를 넣고 [점 찍기]를 누르세요. 클릭으로만 모든 걸 해결하려는 욕망은 아름답지만, 수업 현장에서는 입력칸이 덜 배신합니다.")

cols = st.columns(3)

for i, name in enumerate(stations.keys()):
    with cols[i]:
        st.markdown(f"### {name}")
        x = st.number_input(f"{name} 위도 x", value=stations[name]["lat"], step=0.01, key=f"x_{name}")
        y = st.number_input(f"{name} 경도 y", value=stations[name]["lon"], step=0.01, key=f"y_{name}")

        if st.button(f"{name} 점 찍기", key=f"plot_{name}"):
            st.session_state.student_points[name] = {"lat": x, "lon": y}

# -----------------------------
# 6. 진원거리 입력
# -----------------------------
st.subheader("2단계. 관측소 점에 마우스를 올려 PS시와 진원거리 확인하기")

fig = go.Figure()

for name, s in st.session_state.student_points.items():
    real = stations[name]

    fig.add_trace(go.Scatter(
        x=[s["lat"]],
        y=[s["lon"]],
        mode="markers+text",
        text=[name],
        textposition="top center",
        marker=dict(size=14),
        name=name,
        hovertemplate=(
            f"<b>{name}</b><br>"
            f"PS시: {real['ps_time']}초<br>"
            f"진원거리: {real['distance_km']} km<br>"
            f"반지름 환산: {real['radius_degree']:.3f} 도"
            "<extra></extra>"
        )
    ))

add_base_layout(fig)
st.plotly_chart(fig, use_container_width=True)

distance_inputs = {}

if len(st.session_state.student_points) == 3:
    st.markdown("### 관측소별 진원거리 입력")

    dcols = st.columns(3)
    for i, name in enumerate(stations.keys()):
        with dcols[i]:
            distance_inputs[name] = st.number_input(
                f"{name} 진원거리 km",
                min_value=0.0,
                step=1.0,
                key=f"dist_{name}"
            )

    st.divider()

    # -----------------------------
    # 7. 원의 방정식 입력
    # -----------------------------
    st.subheader("3단계. 원의 방정식 입력하기")

    st.markdown("""
입력 형식 예시:

`(x-37.57)^2 + (y-126.98)^2 = 1.23^2`

주의: 여기서 반지름은 km가 아니라 좌표평면용 도 단위입니다.  
대략 `진원거리 km ÷ 111`로 바꾸면 됩니다. 인간이 지구를 평면으로 펴는 순간 생기는 대가죠 🧭
""")

    for name in stations.keys():
        s = stations[name]
        default_r = s["distance_km"] / KM_PER_DEGREE
        default_eq = f"(x-{s['lat']})^2 + (y-{s['lon']})^2 = {default_r:.3f}^2"

        eq = st.text_input(
            f"{name} 원의 방정식",
            value=default_eq,
            key=f"eq_{name}"
        )

        parsed = parse_circle_equation(eq)

        if parsed:
            st.session_state.circles[name] = parsed
        else:
            st.warning(f"{name}의 원의 방정식 형식이 맞지 않습니다.")

# -----------------------------
# 8. 원, 교점, 현, 진원 표시
# -----------------------------
if len(st.session_state.circles) == 3:
    st.subheader("4단계. 세 원의 교점과 현 확인하기")

    fig2 = go.Figure()

    # 관측소 점
    for name, s in stations.items():
        fig2.add_trace(go.Scatter(
            x=[s["lat"]],
            y=[s["lon"]],
            mode="markers+text",
            text=[name],
            textposition="top center",
            marker=dict(size=12),
            name=name
        ))

    # 원
    circle_list = []
    names = list(st.session_state.circles.keys())

    for name in names:
        h, k, r = st.session_state.circles[name]
        circle_list.append((h, k, r))
        cx, cy = circle_points(h, k, r)

        fig2.add_trace(go.Scatter(
            x=cx,
            y=cy,
            mode="lines",
            name=f"{name} 원"
        ))

    # 원끼리의 교점
    pair_intersections = []
    radical_lines = []

    for i in range(3):
        for j in range(i + 1, 3):
            c1 = circle_list[i]
            c2 = circle_list[j]

            pts = circle_intersections(c1, c2)
            pair_intersections.extend(pts)

            line = radical_axis(c1, c2)
            radical_lines.append(line)

            if len(pts) == 2:
                x_vals = [pts[0][0], pts[1][0]]
                y_vals = [pts[0][1], pts[1][1]]

                fig2.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    name=f"{names[i]}-{names[j]} 현"
                ))

    if pair_intersections:
        fig2.add_trace(go.Scatter(
            x=[p[0] for p in pair_intersections],
            y=[p[1] for p in pair_intersections],
            mode="markers",
            marker=dict(size=10, symbol="x"),
            name="원들의 교점"
        ))

    # 세 현의 교점, 즉 근축의 교점
    estimated = None
    if len(radical_lines) >= 2:
        estimated = line_intersection(radical_lines[0], radical_lines[1])

    if estimated:
        ex, ey = estimated
        fig2.add_trace(go.Scatter(
            x=[ex],
            y=[ey],
            mode="markers+text",
            text=["추정 진원"],
            textposition="bottom center",
            marker=dict(size=16, symbol="star"),
            name="추정 진원"
        ))

    if st.session_state.show_answer:
        fig2.add_trace(go.Scatter(
            x=[actual_epicenter["lat"]],
            y=[actual_epicenter["lon"]],
            mode="markers+text",
            text=["실제 진원"],
            textposition="top center",
            marker=dict(size=18, symbol="diamond"),
            name="실제 진원"
        ))

    add_base_layout(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    if estimated:
        st.success(f"추정 진원 위치: 위도 {estimated[0]:.3f}, 경도 {estimated[1]:.3f}")

    st.divider()

    # -----------------------------
    # 9. 학생 진원 입력 후 실제 결과 확인
    # -----------------------------
    st.subheader("5단계. 진원 위치 추측하기")

    guess_lat = st.number_input("추측한 진원 위도 x", step=0.001, format="%.3f")
    guess_lon = st.number_input("추측한 진원 경도 y", step=0.001, format="%.3f")

    if guess_lat != 0.0 and guess_lon != 0.0:
        if st.button("실제결과 확인"):
            st.session_state.show_answer = True

            error = np.sqrt(
                (guess_lat - actual_epicenter["lat"]) ** 2
                + (guess_lon - actual_epicenter["lon"]) ** 2
            ) * KM_PER_DEGREE

            if error <= 20:
                st.success(f"정확합니다! 실제 진원과의 오차는 약 {error:.1f} km입니다.")
            else:
                st.error(f"조금 차이가 있습니다. 실제 진원과의 오차는 약 {error:.1f} km입니다.")

            st.info(
                f"실제 진원: 위도 {actual_epicenter['lat']}, "
                f"경도 {actual_epicenter['lon']}"
            )

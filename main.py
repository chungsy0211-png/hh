import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ---------------------------------------
# 기본 설정
# ---------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연평균 기온 데이터를 이용해 선형회귀 모델을 만들고 "
    "연도별 예상 기온을 확인합니다."
)

# ---------------------------------------
# 데이터 주소
# ---------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

# ---------------------------------------
# 데이터 불러오기
# ---------------------------------------
@st.cache_data
def load_data():

    # UTF-8 BOM까지 안전하게 처리
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜를 날짜형으로 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온을 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 2025년까지의 데이터만 사용
    df = df[df["연도"] <= 2025].copy()

    # -----------------------------------
    # 연도별 평균기온 + 관측일 수
    # -----------------------------------
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 해 제외
    yearly = yearly[
        yearly["관측일수"] >= 300
    ].copy()

    # 결측값 제거
    yearly = yearly.dropna(
        subset=["연도", "연평균기온"]
    )

    yearly = yearly.sort_values("연도")
    yearly = yearly.reset_index(drop=True)

    return yearly


df = load_data()

# ---------------------------------------
# 회귀분석
# ---------------------------------------
# 1908년부터 지난 연수
df["지난연수"] = df["연도"] - 1908

X = df[["지난연수"]]
y = df["연평균기온"]

model = LinearRegression()
model.fit(X, y)

# 회귀선 예측값
df["회귀예측기온"] = model.predict(X)

# 기울기
slope = model.coef_[0]

# 절편
intercept = model.intercept_

# 상관계수
correlation = df["지난연수"].corr(
    df["연평균기온"]
)

# 결정계수
r2 = model.score(X, y)

# ---------------------------------------
# 기본 정보
# ---------------------------------------
st.subheader("📊 분석 데이터")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "회귀에 사용한 연도 수",
        f"{len(df)}개"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{int(df['연도'].min())}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{int(df['연도'].max())}년"
    )

with col4:
    st.metric(
        "상관계수",
        f"{correlation:.3f}"
    )

st.write(
    f"관측일이 300일 이상이고 2025년 이하인 연도만 사용했습니다. "
    f"회귀선에는 총 **{len(df)}개 연도**가 사용되었습니다."
)

# ---------------------------------------
# 회귀식
# ---------------------------------------
st.subheader("📐 선형회귀 결과")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "100년에 기온 변화",
        f"{slope * 100:+.2f} ℃"
    )

with c2:
    st.metric(
        "회귀선 기울기",
        f"{slope:+.4f} ℃/년"
    )

with c3:
    st.metric(
        "R²",
        f"{r2:.3f}"
    )

st.write(
    f"회귀식: **연평균기온 = {slope:.4f} × (연도 - 1908) "
    f"+ {intercept:.4f}**"
)

# ---------------------------------------
# 산점도 + 회귀선
# ---------------------------------------
st.subheader("📈 연도별 연평균기온과 회귀선")

fig = go.Figure()

# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=df["연도"],
        y=df["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=np.column_stack([
            df["관측일수"],
            df["지난연수"]
        ]),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일<br>"
            "1908년부터 지난 연수: %{customdata[1]}년"
            "<extra></extra>"
        )
    )
)

# 회귀선용 연도
line_years = np.arange(
    df["연도"].min(),
    2026
)

line_x = line_years - 1908

line_y = model.predict(
    pd.DataFrame({
        "지난연수": line_x
    })
)

# 회귀선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y,
        mode="lines",
        name="선형회귀선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    height=600
)

# 가로축은 연도를 그대로 표시
fig.update_xaxes(
    tickmode="auto",
    dtick=10
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------
# 연도 슬라이더
# ---------------------------------------
st.subheader("🔮 원하는 연도의 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 선택한 연도의 지난 연수
selected_x = np.array([[selected_year - 1908]])

# 예측
predicted_temp = model.predict(
    selected_x
)[0]

st.markdown(
    f"""
    <div style="
        text-align:center;
        padding:25px;
        border-radius:15px;
        background-color:#f0f2f6;
        margin-top:10px;
        margin-bottom:20px;
    ">
        <div style="font-size:24px;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="
            font-size:52px;
            font-weight:bold;
            margin-top:10px;
        ">
            {predicted_temp:.2f} ℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------
# 선택한 연도의 회귀 위치 표시
# ---------------------------------------
st.subheader("📍 선택한 연도의 회귀선 위치")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=df["연도"],
        y=df["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        opacity=0.6
    )
)

fig2.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y,
        mode="lines",
        name="회귀선",
        line=dict(width=3)
    )
)

fig2.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=16),
        hovertemplate=(
            f"{selected_year}년<br>"
            "예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=500,
    hovermode="x unified"
)

fig2.update_xaxes(
    tickmode="auto",
    dtick=10
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ---------------------------------------
# 데이터 표
# ---------------------------------------
with st.expander("📋 회귀에 사용된 연도별 데이터 보기"):

    show_df = df[
        [
            "연도",
            "관측일수",
            "연평균기온",
            "지난연수",
            "회귀예측기온"
        ]
    ].copy()

    show_df = show_df.rename(
        columns={
            "연도": "연도",
            "관측일수": "관측일수",
            "연평균기온": "연평균기온(℃)",
            "지난연수": "1908년부터 지난 연수",
            "회귀예측기온": "회귀 예측기온(℃)"
        }
    )

    st.dataframe(
        show_df.style.format({
            "연평균기온(℃)": "{:.2f}",
            "회귀 예측기온(℃)": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------
# 설명
# ---------------------------------------
st.info(
    "주의: 이 앱의 예상 기온은 과거 연평균기온의 선형적인 추세를 "
    "연장한 값입니다. 실제 미래 기온은 여러 기후 요인의 영향을 "
    "받으므로 회귀선의 값을 실제 미래 기온으로 단정할 수는 없습니다."
)

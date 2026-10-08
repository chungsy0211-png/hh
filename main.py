import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균기온을 이용해 1차, 3차, 9차 곡선을 만들고 "
    "학습에 사용하지 않은 테스트 데이터로 성능을 비교합니다."
)


# =========================================================
# 데이터 주소
# =========================================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# =========================================================
# 데이터 불러오기
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 2025년 이후 데이터 제거
    df = df[
        df["연도"] <= 2025
    ].copy()

    # -----------------------------------------------------
    # 연도별 평균기온과 관측일 수 계산
    # -----------------------------------------------------
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일 300일 미만인 해 제거
    yearly = yearly[
        yearly["관측일수"] >= 300
    ].copy()

    # 결측값 제거
    yearly = yearly.dropna(
        subset=["연도", "연평균기온"]
    )

    yearly = yearly.sort_values(
        "연도"
    ).reset_index(drop=True)

    return yearly


df = load_data()


# =========================================================
# 훈련 / 테스트 데이터 분리
# =========================================================
# 중요:
# 2005년 이전 = 훈련 데이터
# 2005년부터 = 테스트 데이터
#
# 즉, 2005년은 테스트 데이터에 포함된다.
# =========================================================

train = df[
    df["연도"] < 2005
].copy()

test = df[
    df["연도"] >= 2005
].copy()


# =========================================================
# 연도 스케일 조정
# =========================================================
# 고차 다항식에서 1900 같은 큰 연도를 그대로 사용하면
# x^9 등의 값이 매우 커져 수치적으로 불안정할 수 있다.
#
# 따라서 가장 이른 연도를 0으로 만들고
# "기준 연도로부터 몇 년 지났는가"를 사용한다.
# =========================================================

base_year = int(df["연도"].min())

train["x"] = train["연도"] - base_year
test["x"] = test["연도"] - base_year


# =========================================================
# 모델 만들기
# =========================================================
def make_model(degree):

    return make_pipeline(
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        LinearRegression()
    )


model_1 = make_model(1)
model_3 = make_model(3)
model_9 = make_model(9)


# =========================================================
# 학습
# =========================================================
X_train = train[["x"]]
y_train = train["연평균기온"]

model_1.fit(X_train, y_train)
model_3.fit(X_train, y_train)
model_9.fit(X_train, y_train)


# =========================================================
# 테스트 데이터 예측
# =========================================================
X_test = test[["x"]]
y_test = test["연평균기온"]

pred_1 = model_1.predict(X_test)
pred_3 = model_3.predict(X_test)
pred_9 = model_9.predict(X_test)


# =========================================================
# 테스트 MAE
# =========================================================
# 반드시 학습에 사용하지 않은 test 데이터만 사용
# =========================================================

mae_1 = mean_absolute_error(
    y_test,
    pred_1
)

mae_3 = mean_absolute_error(
    y_test,
    pred_3
)

mae_9 = mean_absolute_error(
    y_test,
    pred_9
)


# =========================================================
# 2050년 예측
# =========================================================
year_2050 = 2050
x_2050 = np.array([
    [year_2050 - base_year]
])

prediction_2050_1 = model_1.predict(
    x_2050
)[0]

prediction_2050_3 = model_3.predict(
    x_2050
)[0]

prediction_2050_9 = model_9.predict(
    x_2050
)[0]


# =========================================================
# 데이터 개수 표시
# =========================================================
st.subheader("📊 훈련 데이터와 테스트 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 연도 수",
        f"{len(df)}개"
    )

with col2:
    st.metric(
        "훈련 연도 수",
        f"{len(train)}개"
    )

with col3:
    st.metric(
        "테스트 연도 수",
        f"{len(test)}개"
    )


st.write(
    f"""
    - **훈련 데이터:** {int(train["연도"].min())}~{int(train["연도"].max())}년
    - **테스트 데이터:** {int(test["연도"].min())}~{int(test["연도"].max())}년
    - 관측일이 300일 미만인 연도와 2025년 이후 데이터는 제외했습니다.
    """
)

st.warning(
    "테스트 데이터는 모델을 학습할 때 사용하지 않았습니다. "
    "1차·3차·9차 모델 모두 훈련 데이터로만 학습한 뒤 "
    "테스트 데이터에서 MAE를 계산했습니다."
)


# =========================================================
# 결과 표
# =========================================================
st.subheader("📋 곡선별 예측 성능과 2050년 예측")

result = pd.DataFrame({
    "모델": [
        "1차 (직선)",
        "3차 곡선",
        "9차 곡선"
    ],
    "훈련 데이터": [
        f"{int(train['연도'].min())}~{int(train['연도'].max())}",
        f"{int(train['연도'].min())}~{int(train['연도'].max())}",
        f"{int(train['연도'].min())}~{int(train['연도'].max())}"
    ],
    "테스트 MAE (℃)": [
        mae_1,
        mae_3,
        mae_9
    ],
    "2050년 예측 (℃)": [
        prediction_2050_1,
        prediction_2050_3,
        prediction_2050_9
    ]
})

st.dataframe(
    result.style.format({
        "테스트 MAE (℃)": "{:.3f}",
        "2050년 예측 (℃)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 가장 좋은 모델
# =========================================================
mae_values = {
    "1차 (직선)": mae_1,
    "3차 곡선": mae_3,
    "9차 곡선": mae_9
}

best_model = min(
    mae_values,
    key=mae_values.get
)

st.success(
    f"테스트 데이터에서 평균 오차(MAE)가 가장 작은 모델은 "
    f"**{best_model}**입니다. "
    f"MAE는 **{mae_values[best_model]:.3f}℃**입니다."
)


# =========================================================
# 그래프용 예측 데이터
# =========================================================
plot_years = np.arange(
    int(df["연도"].min()),
    2051
)

plot_x = plot_years - base_year

plot_X = pd.DataFrame({
    "x": plot_x
})

plot_pred_1 = model_1.predict(plot_X)
plot_pred_3 = model_3.predict(plot_X)
plot_pred_9 = model_9.predict(plot_X)


# =========================================================
# 그래프
# =========================================================
st.subheader("📈 훈련 데이터와 테스트 데이터, 세 가지 곡선")

fig = go.Figure()


# ---------------------------------------------------------
# 훈련 데이터
# ---------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["연평균기온"],
        mode="markers",
        name="훈련 데이터",
        marker=dict(
            size=6
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# ---------------------------------------------------------
# 테스트 데이터
# ---------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="markers",
        name="테스트 데이터",
        marker=dict(
            size=7
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "실제 연평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# ---------------------------------------------------------
# 1차
# ---------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=plot_years,
        y=plot_pred_1,
        mode="lines",
        name="1차 (직선)",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "1차 예측: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# ---------------------------------------------------------
# 3차
# ---------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=plot_years,
        y=plot_pred_3,
        mode="lines",
        name="3차 곡선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "3차 예측: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# ---------------------------------------------------------
# 9차
# ---------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=plot_years,
        y=plot_pred_9,
        mode="lines",
        name="9차 곡선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "9차 예측: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# ---------------------------------------------------------
# 훈련 / 테스트 구분선
# ---------------------------------------------------------
fig.add_vline(
    x=2005,
    line_dash="dash",
    line_width=2
)

fig.add_annotation(
    x=2005,
    y=1,
    yref="paper",
    text="2005년부터 테스트",
    showarrow=False,
    yanchor="top"
)


# ---------------------------------------------------------
# 그래프 설정
# ---------------------------------------------------------
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=650,
    hovermode="x unified"
)

fig.update_xaxes(
    tickmode="auto",
    dtick=10
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 테스트 데이터의 실제값과 예측값
# =========================================================
st.subheader("🔍 테스트 데이터에서 실제값과 예측값 비교")

comparison = test[
    ["연도", "연평균기온"]
].copy()

comparison["1차 예측"] = pred_1
comparison["3차 예측"] = pred_3
comparison["9차 예측"] = pred_9

comparison["1차 오차"] = (
    comparison["1차 예측"]
    - comparison["연평균기온"]
)

comparison["3차 오차"] = (
    comparison["3차 예측"]
    - comparison["연평균기온"]
)

comparison["9차 오차"] = (
    comparison["9차 예측"]
    - comparison["연평균기온"]
)

st.dataframe(
    comparison.style.format({
        "연평균기온": "{:.2f}",
        "1차 예측": "{:.2f}",
        "3차 예측": "{:.2f}",
        "9차 예측": "{:.2f}",
        "1차 오차": "{:+.2f}",
        "3차 오차": "{:+.2f}",
        "9차 오차": "{:+.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 설명
# =========================================================
st.subheader("💡 해석하기")

st.write(
    """
    **MAE(평균 절대 오차)**는 실제 기온과 예측 기온이 평균적으로
    몇 ℃ 차이 나는지를 나타냅니다. 따라서 값이 작을수록
    테스트 데이터를 잘 예측한 모델입니다.

    이번 분석에서는 2005년 이전의 데이터만으로 곡선을 학습하고,
    2005년 이후의 데이터는 학습에 전혀 사용하지 않은 채
    마지막에 성능을 평가했습니다.

    특히 9차 곡선은 훈련 데이터에서는 매우 복잡한 모양을 만들 수 있지만,
    테스트 데이터에서도 좋은 성능을 내는지는 별개의 문제입니다.
    따라서 단순히 훈련 데이터에 잘 맞는 곡선을 고르는 것이 아니라
    **처음 보지 않은 테스트 데이터의 MAE가 작은 모델**을 비교해야 합니다.

    또한 고차 곡선의 계산이 불안정해지는 것을 막기 위해
    실제 연도(예: 1908)를 그대로 거듭제곱하지 않고
    **가장 이른 연도부터 몇 년이 지났는지**로 변환하여 계산했습니다.
    """
)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------
# 기본 설정
# -----------------------------------------
st.set_page_config(
    page_title="서울 기온 선형회귀 평가",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 모델 평가")

st.write(
    "과거의 연평균 기온으로 선형회귀 모델을 학습하고, "
    "최근 20년의 기온을 얼마나 잘 예측하는지 비교합니다."
)

# -----------------------------------------
# 데이터 주소
# -----------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

# -----------------------------------------
# 데이터 불러오기 및 연평균 계산
# -----------------------------------------
@st.cache_data
def load_yearly_data():
    df = pd.read_csv(DATA_URL)

    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 연도별 평균기온과 관측일 수
    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(["mean", "count"])
        .reset_index()
    )

    yearly = yearly.rename(
        columns={"mean": "연평균기온", "count": "관측일수"}
    )

    # 2025년까지 + 관측일 300일 이상
    yearly = yearly[
        (yearly["연도"] <= 2025) &
        (yearly["관측일수"] >= 300)
    ].copy()

    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


yearly = load_yearly_data()

# -----------------------------------------
# 분석 기간
# -----------------------------------------
TEST_START = 2006
TEST_END = 2025

TRAIN_50_START = 1956
TRAIN_END = 2005

TRAIN_100_START = 1906

# 테스트 데이터
test = yearly[
    (yearly["연도"] >= TEST_START) &
    (yearly["연도"] <= TEST_END)
].copy()

# 최근 50년 훈련 데이터
train_50 = yearly[
    (yearly["연도"] >= TRAIN_50_START) &
    (yearly["연도"] <= TRAIN_END)
].copy()

# 최근 100년 훈련 데이터
train_100 = yearly[
    (yearly["연도"] >= TRAIN_100_START) &
    (yearly["연도"] <= TRAIN_END)
].copy()

# -----------------------------------------
# 전체 데이터 모델
# -----------------------------------------
overall = yearly[
    (yearly["연도"] >= yearly["연도"].min()) &
    (yearly["연도"] <= 2025)
].copy()

X_overall = overall[["연도"]]
y_overall = overall["연평균기온"]

overall_model = LinearRegression()
overall_model.fit(X_overall, y_overall)

overall_pred = overall_model.predict(X_overall)

overall_mae = mean_absolute_error(y_overall, overall_pred)
overall_mse = mean_squared_error(y_overall, overall_pred)
overall_r2 = r2_score(y_overall, overall_pred)

overall_slope = overall_model.coef_[0]

# -----------------------------------------
# 50년 모델
# -----------------------------------------
X_train_50 = train_50[["연도"]]
y_train_50 = train_50["연평균기온"]

model_50 = LinearRegression()
model_50.fit(X_train_50, y_train_50)

pred_50 = model_50.predict(test[["연도"]])

mae_50 = mean_absolute_error(test["연평균기온"], pred_50)
mse_50 = mean_squared_error(test["연평균기온"], pred_50)
r2_50 = r2_score(test["연평균기온"], pred_50)

slope_50 = model_50.coef_[0]

# -----------------------------------------
# 100년 모델
# -----------------------------------------
X_train_100 = train_100[["연도"]]
y_train_100 = train_100["연평균기온"]

model_100 = LinearRegression()
model_100.fit(X_train_100, y_train_100)

pred_100 = model_100.predict(test[["연도"]])

mae_100 = mean_absolute_error(test["연평균기온"], pred_100)
mse_100 = mean_squared_error(test["연평균기온"], pred_100)
r2_100 = r2_score(test["연평균기온"], pred_100)

slope_100 = model_100.coef_[0]

# -----------------------------------------
# 데이터 기간 정보
# -----------------------------------------
st.subheader("📌 데이터 구성")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "전체 연평균 데이터",
        f"{len(yearly)}개 연도"
    )

with c2:
    st.metric(
        "훈련 데이터",
        "1956~2005 / 1906~2005"
    )

with c3:
    st.metric(
        "공통 테스트 데이터",
        "2006~2025"
    )

st.info(
    f"실제 사용 가능한 연평균 데이터는 "
    f"{yearly['연도'].min()}~{yearly['연도'].max()}년입니다. "
    f"따라서 1906~1907년에 유효한 연평균 데이터가 없다면 "
    f"100년 훈련 모델은 사용 가능한 데이터부터 학습합니다."
)

# -----------------------------------------
# 전체 데이터 평가
# -----------------------------------------
st.subheader("1️⃣ 지난번 전체 데이터로 만든 회귀선 평가")

st.write(
    "전체 유효 연평균 데이터를 하나의 회귀 모델로 학습했을 때의 "
    "훈련 데이터 기준 성능입니다."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "기울기",
        f"{overall_slope * 100:+.2f} ℃ / 100년"
    )

with col2:
    st.metric(
        "MAE",
        f"{overall_mae:.3f} ℃"
    )

with col3:
    st.metric(
        "MSE",
        f"{overall_mse:.3f}"
    )

with col4:
    st.metric(
        "R²",
        f"{overall_r2:.3f}"
    )

# -----------------------------------------
# 훈련 / 테스트 데이터 설명
# -----------------------------------------
st.subheader("2️⃣ 훈련 데이터와 테스트 데이터")

st.write(
    "두 모델 모두 **2006~2025년을 공통 테스트 데이터**로 사용합니다. "
    "따라서 어떤 모델이 최근 20년을 더 잘 예측했는지 공정하게 비교할 수 있습니다."
)

split_df = pd.DataFrame({
    "구분": [
        "최근 50년 모델 훈련",
        "최근 100년 모델 훈련",
        "공통 테스트"
    ],
    "기간": [
        "1956~2005",
        "1906~2005",
        "2006~2025"
    ],
    "사용 연도 수": [
        len(train_50),
        len(train_100),
        len(test)
    ]
})

st.dataframe(
    split_df,
    use_container_width=True,
    hide_index=True
)

# -----------------------------------------
# 회귀선 비교 그래프
# -----------------------------------------
st.subheader("3️⃣ 50년 학습 vs 100년 학습 회귀선")

fig = go.Figure()

# 전체 실제 데이터
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        opacity=0.6
    )
)

# 50년 훈련 회귀선
x_line_50 = np.arange(
    TRAIN_50_START,
    TEST_END + 1
)

y_line_50 = model_50.predict(
    pd.DataFrame({"연도": x_line_50})
)

fig.add_trace(
    go.Scatter(
        x=x_line_50,
        y=y_line_50,
        mode="lines",
        name="1956~2005 학습 회귀선"
    )
)

# 100년 훈련 회귀선
x_line_100 = np.arange(
    train_100["연도"].min(),
    TEST_END + 1
)

y_line_100 = model_100.predict(
    pd.DataFrame({"연도": x_line_100})
)

fig.add_trace(
    go.Scatter(
        x=x_line_100,
        y=y_line_100,
        mode="lines",
        name="1906~2005 학습 회귀선"
    )
)

# 테스트 기간 표시
fig.add_vrect(
    x0=TEST_START,
    x1=TEST_END,
    fillcolor="gray",
    opacity=0.15,
    line_width=0,
    annotation_text="공통 테스트 기간"
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified",
    height=550
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------
# 기울기 비교
# -----------------------------------------
st.subheader("4️⃣ 회귀선의 기울기 비교")

slope_compare = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "훈련 기간": [
        "1956~2005",
        "1906~2005"
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100
    ],
    "100년당 기온 변화 (℃)": [
        slope_50 * 100,
        slope_100 * 100
    ]
})

st.dataframe(
    slope_compare.style.format({
        "기울기 (℃/년)": "{:+.4f}",
        "100년당 기온 변화 (℃)": "{:+.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "최근 50년 회귀선",
        f"{slope_50 * 100:+.2f} ℃ / 100년"
    )

with col2:
    st.metric(
        "최근 100년 회귀선",
        f"{slope_100 * 100:+.2f} ℃ / 100년"
    )

# -----------------------------------------
# 테스트 성능 비교
# -----------------------------------------
st.subheader("5️⃣ 최근 20년(2006~2025) 예측 성능 비교")

st.write(
    "두 모델 모두 한 번도 학습에 사용하지 않은 2006~2025년 데이터를 "
    "예측한 결과입니다. MAE와 MSE는 작을수록 좋고, R²는 클수록 좋습니다."
)

performance = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "훈련 기간": [
        "1956~2005",
        "1906~2005"
    ],
    "MAE (℃)": [
        mae_50,
        mae_100
    ],
    "MSE": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    performance.style.format({
        "MAE (℃)": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

# -----------------------------------------
# 성능을 카드로 비교
# -----------------------------------------
st.markdown("### 📊 모델별 테스트 성능")

a, b = st.columns(2)

with a:
    st.markdown("#### 🟦 최근 50년 학습")
    st.metric("MAE", f"{mae_50:.3f} ℃")
    st.metric("MSE", f"{mse_50:.3f}")
    st.metric("R²", f"{r2_50:.3f}")

with b:
    st.markdown("#### 🟩 최근 100년 학습")
    st.metric("MAE", f"{mae_100:.3f} ℃")
    st.metric("MSE", f"{mse_100:.3f}")
    st.metric("R²", f"{r2_100:.3f}")

# -----------------------------------------
# 테스트 실제값 vs 예측값
# -----------------------------------------
st.subheader("6️⃣ 2006~2025년 실제 기온과 예측 기온 비교")

test_plot = test.copy()

test_plot["50년 모델 예측"] = pred_50
test_plot["100년 모델 예측"] = pred_100

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=test_plot["연도"],
        y=test_plot["연평균기온"],
        mode="lines+markers",
        name="실제 기온"
    )
)

fig2.add_trace(
    go.Scatter(
        x=test_plot["연도"],
        y=test_plot["50년 모델 예측"],
        mode="lines",
        name="50년 학습 예측"
    )
)

fig2.add_trace(
    go.Scatter(
        x=test_plot["연도"],
        y=test_plot["100년 모델 예측"],
        mode="lines",
        name="100년 학습 예측"
    )
)

fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified",
    height=500
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# -----------------------------------------
# 어떤 모델이 더 좋은가?
# -----------------------------------------
st.subheader("7️⃣ 결과 해석")

mae_winner = "최근 50년" if mae_50 < mae_100 else "최근 100년"
mse_winner = "최근 50년" if mse_50 < mse_100 else "최근 100년"
r2_winner = "최근 50년" if r2_50 > r2_100 else "최근 100년"

st.write(
    f"""
- **MAE가 더 작은 모델:** {mae_winner}
- **MSE가 더 작은 모델:** {mse_winner}
- **R²가 더 큰 모델:** {r2_winner}

MAE는 실제 기온과 예측 기온의 평균적인 차이를 나타내므로 작을수록 좋습니다.
MSE는 큰 오차에 더 큰 불이익을 주기 때문에 역시 작을수록 좋습니다.
R²는 1에 가까울수록 테스트 데이터의 변동을 잘 설명한다는 뜻입니다.
"""
)

st.info(
    "중요: R²가 음수가 나올 수도 있습니다. "
    "이는 테스트 데이터에서 모델이 단순히 평균값을 사용하는 것보다도 "
    "예측을 잘하지 못했다는 뜻입니다."
)

# -----------------------------------------
# 결론
# -----------------------------------------
st.subheader("📝 탐구 결론을 정리할 때")

st.write(
    f"""
이번 분석에서는 과거 기온을 이용해 두 개의 선형회귀 모델을 만들고,
동일한 2006~2025년 데이터를 테스트 데이터로 사용하여 비교했습니다.

최근 50년을 학습한 회귀선의 기울기는
**{slope_50 * 100:+.2f}℃/100년**이고,

최근 100년을 학습한 회귀선의 기울기는
**{slope_100 * 100:+.2f}℃/100년**입니다.

따라서 두 학습 기간을 선택했을 때 기울기가 서로 다르게 나타나는지,
그리고 과거 데이터를 더 많이 사용할수록 최근 20년의 기온을
더 잘 예측하는지 비교할 수 있습니다.
"""
)

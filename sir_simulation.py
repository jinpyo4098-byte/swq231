import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="전염병 확산 시뮬레이션 (SIR 모델)",
    page_icon="🦠",
    layout="wide",
)

# ── CSS 스타일 ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&family=Space+Mono&display=swap');

  html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

  .main-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #e74c3c, #e67e22, #f39c12);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .subtitle { color: #7f8c8d; font-size: 1rem; margin-bottom: 1.5rem; }

  .metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 12px; padding: 1rem 1.2rem;
    border-left: 4px solid;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
  }
  .metric-s { border-color: #3498db; }
  .metric-i { border-color: #e74c3c; }
  .metric-r { border-color: #2ecc71; }
  .metric-label { font-size: 0.75rem; color: #bdc3c7; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-size: 2rem; font-weight: 700; font-family: 'Space Mono', monospace; }
  .metric-s .metric-value { color: #3498db; }
  .metric-i .metric-value { color: #e74c3c; }
  .metric-r .metric-value { color: #2ecc71; }

  .info-box {
    background: #1e2a3a; border-radius: 10px; padding: 1rem 1.3rem;
    border: 1px solid #2c3e50; margin: 0.5rem 0;
    font-size: 0.9rem; color: #ecf0f1;
  }
  .info-box strong { color: #f39c12; }

  .r0-badge {
    display: inline-block;
    padding: 0.3rem 1rem; border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem; font-weight: 700;
  }
  .r0-danger  { background: #c0392b33; color: #e74c3c; border: 1px solid #e74c3c; }
  .r0-warning { background: #e67e2233; color: #e67e22; border: 1px solid #e67e22; }
  .r0-safe    { background: #27ae6033; color: #2ecc71; border: 1px solid #2ecc71; }
</style>
""", unsafe_allow_html=True)


# ── SIR / SIRV 모델 ──────────────────────────────────────────────────────────
def sir_model(y, t, beta, gamma):
    """기본 SIR 미분방정식"""
    S, I, R = y
    N = S + I + R
    dS = -beta * S * I / N
    dI =  beta * S * I / N - gamma * I
    dR =  gamma * I
    return [dS, dI, dR]


def sirv_model(y, t, beta, gamma, v_rate, social_dist):
    """백신 접종 + 거리두기 확장 모델 (SIRV)"""
    S, I, R, V = y
    N = S + I + R + V
    beta_eff = beta * (1 - social_dist)   # 거리두기로 감소한 전파율
    dS = -beta_eff * S * I / N - v_rate * S
    dI =  beta_eff * S * I / N - gamma * I
    dR =  gamma * I
    dV =  v_rate * S                       # 백신 접종 구획
    return [dS, dI, dR, dV]


def run_simulation(N, I0, beta, gamma, v_rate, social_dist, days):
    t = np.linspace(0, days, days * 10)
    S0 = N - I0
    R0_init = 0
    V0 = 0
    y0 = [S0, I0, R0_init, V0]
    sol = odeint(sirv_model, y0, t,
                 args=(beta, gamma, v_rate, social_dist))
    S, I, R, V = sol.T
    return t, S, I, R, V


# ── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 시뮬레이션 파라미터")

    st.markdown("### 👥 인구 설정")
    N = st.slider("총 인구 (N)", 1_000, 1_000_000, 100_000, step=1_000,
                  format="%d명")
    I0 = st.slider("초기 감염자 수 (I₀)", 1, 500, 10)

    st.markdown("### 🦠 전염병 특성")
    beta = st.slider("감염률 (β) — 단위 시간당 전파 확률",
                     0.05, 1.0, 0.30, step=0.01)
    gamma = st.slider("회복률 (γ) — 1/평균 감염 기간",
                      0.01, 0.5, 0.10, step=0.01)

    st.markdown("### 💉 방역 조치")
    v_rate = st.slider("일일 백신 접종률 (%)",
                       0.0, 5.0, 1.0, step=0.1) / 100
    social_dist = st.slider("사회적 거리두기 강도 (%)",
                            0, 100, 30, step=5) / 100

    days = st.slider("시뮬레이션 기간 (일)", 30, 365, 180)

    st.markdown("---")
    st.markdown("### 📊 비교 시뮬레이션")
    compare = st.checkbox("방역 없는 시나리오와 비교", value=True)


# ── 메인 콘텐츠 ──────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🦠 전염병 확산 시뮬레이션</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">SIR / SIRV 미분방정식 모델 · 백신 접종 & 사회적 거리두기 효과 분석</p>',
            unsafe_allow_html=True)

# 시뮬레이션 실행
t, S, I, R, V = run_simulation(N, I0, beta, gamma, v_rate, social_dist, days)

# 기초감염재생산수 R₀
R0_eff = beta * (1 - social_dist) / gamma
if R0_eff > 2:
    badge_class, badge_label = "r0-danger", "위험"
elif R0_eff > 1:
    badge_class, badge_label = "r0-warning", "경계"
else:
    badge_class, badge_label = "r0-safe", "통제"

# ── 핵심 지표 ────────────────────────────────────────────────────────────────
peak_infected = int(I.max())
peak_day = int(t[I.argmax()])
total_infected = int(R[-1] + I[-1])
vaccinated = int(V[-1])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card metric-i">
      <div class="metric-label">🔴 최대 동시 감염자</div>
      <div class="metric-value">{peak_infected:,}</div>
      <div style="color:#bdc3c7;font-size:0.8rem">{peak_day}일째 정점</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card metric-r">
      <div class="metric-label">🟢 최종 감염 누적</div>
      <div class="metric-value">{total_infected:,}</div>
      <div style="color:#bdc3c7;font-size:0.8rem">인구의 {total_infected/N*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card metric-s">
      <div class="metric-label">💉 백신 접종 완료</div>
      <div class="metric-value">{vaccinated:,}</div>
      <div style="color:#bdc3c7;font-size:0.8rem">인구의 {vaccinated/N*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card" style="border-color:#f39c12">
      <div class="metric-label">📈 기초감염재생산수 R₀ (유효)</div>
      <div class="metric-value" style="color:#f39c12">{R0_eff:.2f}</div>
      <div><span class="r0-badge {badge_class}">{badge_label} — R₀ {'>' if R0_eff > 1 else '<'} 1</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 메인 그래프 ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 시계열 그래프", "📊 비율 스택 차트", "🔍 비교 분석"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=S, name="감수성자 S (미감염)",
                             line=dict(color="#3498db", width=2.5),
                             fill="tozeroy", fillcolor="rgba(52,152,219,0.08)"))
    fig.add_trace(go.Scatter(x=t, y=I, name="감염자 I",
                             line=dict(color="#e74c3c", width=3),
                             fill="tozeroy", fillcolor="rgba(231,76,60,0.12)"))
    fig.add_trace(go.Scatter(x=t, y=R, name="회복자 R",
                             line=dict(color="#2ecc71", width=2.5),
                             fill="tozeroy", fillcolor="rgba(46,204,113,0.08)"))
    fig.add_trace(go.Scatter(x=t, y=V, name="백신 접종자 V",
                             line=dict(color="#9b59b6", width=2, dash="dash")))

    # 정점 표시
    fig.add_vline(x=peak_day, line_dash="dot", line_color="#e74c3c",
                  annotation_text=f"  정점 {peak_day}일",
                  annotation_font_color="#e74c3c")

    fig.update_layout(
        title=dict(text="SIRV 모델 — 시간에 따른 각 구획 변화",
                   font=dict(size=16, color="#ecf0f1")),
        xaxis=dict(title="경과 일수 (일)", color="#bdc3c7",
                   gridcolor="#2c3e50", showgrid=True),
        yaxis=dict(title="인구 수", color="#bdc3c7",
                   gridcolor="#2c3e50", showgrid=True),
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        legend=dict(bgcolor="#1a1a2e", bordercolor="#2c3e50",
                    font=dict(color="#ecf0f1")),
        hovermode="x unified",
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=S/N*100, name="감수성자 S",
                              stackgroup="one", fillcolor="rgba(52,152,219,0.7)",
                              line=dict(color="#3498db")))
    fig2.add_trace(go.Scatter(x=t, y=I/N*100, name="감염자 I",
                              stackgroup="one", fillcolor="rgba(231,76,60,0.7)",
                              line=dict(color="#e74c3c")))
    fig2.add_trace(go.Scatter(x=t, y=R/N*100, name="회복자 R",
                              stackgroup="one", fillcolor="rgba(46,204,113,0.7)",
                              line=dict(color="#2ecc71")))
    fig2.add_trace(go.Scatter(x=t, y=V/N*100, name="백신 접종자 V",
                              stackgroup="one", fillcolor="rgba(155,89,182,0.7)",
                              line=dict(color="#9b59b6")))
    fig2.update_layout(
        title=dict(text="인구 비율 스택 (%) — 각 구획 비중 변화",
                   font=dict(size=16, color="#ecf0f1")),
        xaxis=dict(title="경과 일수 (일)", color="#bdc3c7", gridcolor="#2c3e50"),
        yaxis=dict(title="인구 비율 (%)", color="#bdc3c7", gridcolor="#2c3e50",
                   range=[0, 100]),
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        legend=dict(bgcolor="#1a1a2e", bordercolor="#2c3e50",
                    font=dict(color="#ecf0f1")),
        height=480,
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    if compare:
        t2, S2, I2, R2, V2 = run_simulation(N, I0, beta, gamma, 0.0, 0.0, days)
        fig3 = make_subplots(rows=1, cols=2,
                             subplot_titles=("방역 없음 (기본 SIR)", "방역 적용 (현재 설정)"))

        for col_idx, (tt, ii, rr, label) in enumerate([
            (t2, I2, R2, "방역 없음"),
            (t,  I,  R,  "방역 적용")
        ], start=1):
            fig3.add_trace(go.Scatter(x=tt, y=ii, name=f"감염자 ({label})",
                                      line=dict(color="#e74c3c", width=2.5),
                                      fill="tozeroy",
                                      fillcolor="rgba(231,76,60,0.15)"),
                           row=1, col=col_idx)
            fig3.add_trace(go.Scatter(x=tt, y=rr, name=f"회복자 ({label})",
                                      line=dict(color="#2ecc71", width=2)),
                           row=1, col=col_idx)

        prevented = int(R2[-1] - total_infected)
        fig3.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=440, showlegend=False,
            title=dict(
                text=f"방역 효과 비교 — 방역으로 약 {prevented:,}명 추가 감염 예방",
                font=dict(size=15, color="#f39c12")
            ),
        )
        fig3.update_xaxes(color="#bdc3c7", gridcolor="#2c3e50")
        fig3.update_yaxes(color="#bdc3c7", gridcolor="#2c3e50")
        st.plotly_chart(fig3, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""<div class="info-box">
              <strong>🚫 방역 없음</strong><br>
              최대 동시 감염자: <strong>{int(I2.max()):,}명</strong><br>
              최종 감염 누적: <strong>{int(R2[-1]):,}명 ({int(R2[-1])/N*100:.1f}%)</strong>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""<div class="info-box">
              <strong>✅ 방역 적용 (현재)</strong><br>
              최대 동시 감염자: <strong>{peak_infected:,}명</strong><br>
              최종 감염 누적: <strong>{total_infected:,}명 ({total_infected/N*100:.1f}%)</strong><br>
              예방 효과: <strong style="color:#2ecc71">약 {prevented:,}명 감소 🎉</strong>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("사이드바에서 '방역 없는 시나리오와 비교' 옵션을 활성화하세요.")

# ── 수식 설명 ────────────────────────────────────────────────────────────────
with st.expander("📐 수학적 모델 설명 (클릭하여 펼치기)"):
    st.markdown(r"""
### SIRV 미분방정식 시스템

$$\frac{dS}{dt} = -\frac{\beta_{\text{eff}} \cdot S \cdot I}{N} - v \cdot S$$

$$\frac{dI}{dt} = \frac{\beta_{\text{eff}} \cdot S \cdot I}{N} - \gamma \cdot I$$

$$\frac{dR}{dt} = \gamma \cdot I$$

$$\frac{dV}{dt} = v \cdot S$$

**유효 감염률:** $\beta_{\text{eff}} = \beta \times (1 - \delta)$  ← 거리두기 강도 δ 반영

**기초감염재생산수:** $R_0 = \dfrac{\beta_{\text{eff}}}{\gamma}$
- $R_0 > 1$ → 전염병 확산
- $R_0 < 1$ → 전염병 소멸

| 파라미터 | 의미 | 현재값 |
|----------|------|--------|
| β (베타) | 단위 시간당 감염 전파 확률 | """ + f"{beta:.2f}" + r""" |
| γ (감마) | 회복률 (= 1 / 평균 감염 기간) | """ + f"{gamma:.2f}" + r""" |
| δ (델타) | 사회적 거리두기 강도 | """ + f"{social_dist:.0%}" + r""" |
| v | 일일 백신 접종률 | """ + f"{v_rate:.2%}" + r""" |
""")

st.markdown("""
<hr style="border-color:#2c3e50; margin-top:2rem">
<div style="text-align:center; color:#7f8c8d; font-size:0.8rem; padding:0.5rem">
  SIR/SIRV Epidemic Simulation · Python (scipy.integrate.odeint) · Plotly · Streamlit
</div>
""", unsafe_allow_html=True)

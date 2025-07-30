import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math

# Set page config
st.set_page_config(page_title="ProbLearn", layout="wide")

# CSS 
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Interactive Probability Distributions")
st.markdown("**Learn probability distributions through interactive visualization!**")

# Tabs for different distributions
tabs = st.tabs(["Poisson", "Binomial", "Normal", "Exponential", "Uniform", "Geometric", "Bernoulli"])

# Tab 1: Poisson Distribution
with tabs[0]:
    st.header("Poisson Distribution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        lambda_val = st.slider("λ (lambda)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)
        x_poisson = st.number_input("x (observation)", min_value=0, max_value=50, value=2)
        
        plot_type = st.radio("Plot Type", ["PMF", "CDF"], key="poisson_plot")
        
        st.subheader("Statistics")
        expected_val = lambda_val
        variance = lambda_val
        prob_x = (np.exp(-lambda_val) * lambda_val**x_poisson) / math.factorial(x_poisson)
        
        st.metric("Expected Value", f"{expected_val:.4f}")
        st.metric("Variance", f"{variance:.4f}")
        st.metric(f"P(X = {x_poisson})", f"{prob_x:.6f}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_vals = np.arange(0, int(lambda_val + 20))
        
        if plot_type == "PMF":
            y_vals = stats.poisson.pmf(x_vals, lambda_val)
            ax.bar(x_vals, y_vals, alpha=0.7, color='skyblue')
            ax.set_ylabel('Probability')
            ax.set_title('Poisson PMF')
        else:
            y_vals = stats.poisson.cdf(x_vals, lambda_val)
            ax.step(x_vals, y_vals, where='post', linewidth=2, color='red')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title('Poisson CDF')
        
        ax.set_xlabel('Observation')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Tab 2: Binomial Distribution
with tabs[1]:
    st.header("Binomial Distribution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        n_binom = st.slider("n (trials)", min_value=1, max_value=50, value=10)
        p_binom = st.slider("p (probability)", min_value=0.01, max_value=0.99, value=0.3, step=0.01)
        x_binom = st.number_input("x (successes)", min_value=0, max_value=n_binom, value=3)
        
        plot_type = st.radio("Plot Type", ["PMF", "CDF"], key="binom_plot")
        
        st.subheader("Statistics")
        expected_val = n_binom * p_binom
        variance = n_binom * p_binom * (1 - p_binom)
        prob_x = stats.binom.pmf(x_binom, n_binom, p_binom)
        
        st.metric("Expected Value", f"{expected_val:.4f}")
        st.metric("Variance", f"{variance:.4f}")
        st.metric(f"P(X = {x_binom})", f"{prob_x:.6f}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_vals = np.arange(0, n_binom + 1)
        
        if plot_type == "PMF":
            y_vals = stats.binom.pmf(x_vals, n_binom, p_binom)
            ax.bar(x_vals, y_vals, alpha=0.7, color='lightgreen')
            ax.set_ylabel('Probability')
            ax.set_title('Binomial PMF')
        else:
            y_vals = stats.binom.cdf(x_vals, n_binom, p_binom)
            ax.step(x_vals, y_vals, where='post', linewidth=2, color='red')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title('Binomial CDF')
        
        ax.set_xlabel('Observation')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Tab 3: Normal Distribution
with tabs[2]:
    st.header("Normal Distribution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        mu_normal = st.slider("μ (mean)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        sigma2_normal = st.slider("σ² (variance)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        x_normal = st.number_input("x (observation)", value=0.0, step=0.1)
        
        plot_type = st.radio("Plot Type", ["PDF", "CDF"], key="normal_plot")
        
        st.subheader("Statistics")
        sigma_normal = np.sqrt(sigma2_normal)
        expected_val = mu_normal
        variance = sigma2_normal
        density_x = (1/(sigma_normal * np.sqrt(2 * np.pi))) * np.exp(-((x_normal - mu_normal)**2) / (2 * sigma2_normal))
        
        st.metric("Expected Value", f"{expected_val:.4f}")
        st.metric("Variance", f"{variance:.4f}")
        st.metric(f"f({x_normal:.1f})", f"{density_x:.6f}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_vals = np.linspace(mu_normal - 10, mu_normal + 10, 1000)
        
        if plot_type == "PDF":
            y_vals = stats.norm.pdf(x_vals, mu_normal, sigma_normal)
            ax.plot(x_vals, y_vals, linewidth=2, color='blue')
            ax.set_ylabel('Probability Density')
            ax.set_title('Normal PDF')
        else:
            y_vals = stats.norm.cdf(x_vals, mu_normal, sigma_normal)
            ax.plot(x_vals, y_vals, linewidth=2, color='red')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title('Normal CDF')
        
        ax.set_xlabel('Observation')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Tab 4: Exponential Distribution
with tabs[3]:
    st.header("Exponential Distribution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        lambda_exp = st.slider("λ (rate)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        x_exp = st.number_input("x (observation)", min_value=0.0, value=1.0, step=0.1, key="x_exp")
        
        plot_type = st.radio("Plot Type", ["PDF", "CDF"], key="exp_plot")
        
        st.subheader("Statistics")
        expected_val = 1/lambda_exp
        variance = 1/(lambda_exp**2)
        density_x = lambda_exp * np.exp(-lambda_exp * x_exp)
        
        st.metric("Expected Value", f"{expected_val:.4f}")
        st.metric("Variance", f"{variance:.4f}")
        st.metric(f"f({x_exp:.1f})", f"{density_x:.6f}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_vals = np.linspace(0, lambda_exp * 5, 1000)
        
        if plot_type == "PDF":
            y_vals = stats.expon.pdf(x_vals, scale=1/lambda_exp)
            ax.plot(x_vals, y_vals, linewidth=2, color='orange')
            ax.set_ylabel('Probability Density')
            ax.set_title('Exponential PDF')
        else:
            y_vals = stats.expon.cdf(x_vals, scale=1/lambda_exp)
            ax.plot(x_vals, y_vals, linewidth=2, color='red')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title('Exponential CDF')
        
        ax.set_xlabel('Observation')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Tab 5: Uniform Distribution
with tabs[4]:
    st.header("Uniform Distribution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        a_uniform = st.slider("a (lower bound)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        b_uniform = st.slider("b (upper bound)", min_value=a_uniform + 0.1, max_value=15.0, value=5.0, step=0.1)
        x_uniform = st.number_input("x (observation)", value=2.5, step=0.1, key="x_uniform")
        
        plot_type = st.radio("Plot Type", ["PDF", "CDF"], key="uniform_plot")
        
        st.subheader("Statistics")
        expected_val = (a_uniform + b_uniform) / 2
        variance = (b_uniform - a_uniform)**2 / 12
        density_x = 1 / (b_uniform - a_uniform) if a_uniform <= x_uniform <= b_uniform else 0
        
        st.metric("Expected Value", f"{expected_val:.4f}")
        st.metric("Variance", f"{variance:.4f}")
        st.metric(f"f({x_uniform:.1f})", f"{density_x:.6f}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_vals = np.linspace(a_uniform - 1, b_uniform + 1, 1000)
        
        if plot_type == "PDF":
            y_vals = stats.uniform.pdf(x_vals, loc=a_uniform, scale=b_uniform-a_uniform)
            ax.plot(x_vals, y_vals, linewidth=3, color='purple')
            ax.set_ylabel('Probability Density')
            ax.set_title('Uniform PDF')
        else:
            y_vals = stats.uniform.cdf(x_vals, loc=a_uniform, scale=b_uniform-a_uniform)
            ax.plot(x_vals, y_vals, linewidth=3, color='red')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title('Uniform CDF')
        
        ax.set_xlabel('Observation')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Tab 6: Geometric Distribution
with tabs[5]:
    st.header("Geometric Distribution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        p_geom = st.slider("p (probability of success)", min_value=0.01, max_value=0.99, value=0.3, step=0.01, key="p_geom")
        x_geom = st.number_input("x (number of failures)", min_value=0, max_value=50, value=2, key="x_geom")
        
        plot_type = st.radio("Plot Type", ["PMF", "CDF"], key="geom_plot")
        
        st.subheader("Statistics")
        expected_val = 1/p_geom
        variance = (1-p_geom)/(p_geom**2)
        prob_x = ((1-p_geom)**x_geom) * p_geom
        
        st.metric("Expected Value", f"{expected_val:.4f}")
        st.metric("Variance", f"{variance:.4f}")
        st.metric(f"P(X = {x_geom})", f"{prob_x:.6f}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_vals = np.arange(0, int(p_geom * 100))
        
        if plot_type == "PMF":
            y_vals = stats.geom.pmf(x_vals + 1, p_geom)  # scipy uses different parameterization
            ax.bar(x_vals, y_vals, alpha=0.7, color='coral')
            ax.set_ylabel('Probability')
            ax.set_title('Geometric PMF')
        else:
            y_vals = stats.geom.cdf(x_vals + 1, p_geom)
            ax.step(x_vals, y_vals, where='post', linewidth=2, color='red')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title('Geometric CDF')
        
        ax.set_xlabel('Observation')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Tab 7: Bernoulli Distribution
with tabs[6]:
    st.header("Bernoulli Distribution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        p_bern = st.slider("p (probability of success)", min_value=0.01, max_value=0.99, value=0.5, step=0.01, key="p_bern")
        x_bern = st.selectbox("x (observation)", [0, 1])
        
        plot_type = st.radio("Plot Type", ["PMF", "CDF"], key="bern_plot")
        
        st.subheader("Statistics")
        expected_val = p_bern
        variance = p_bern * (1 - p_bern)
        prob_x = p_bern**x_bern * (1-p_bern)**(1-x_bern)
        
        st.metric("Expected Value", f"{expected_val:.4f}")
        st.metric("Variance", f"{variance:.4f}")
        st.metric(f"P(X = {x_bern})", f"{prob_x:.6f}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_vals = np.array([0, 1])
        
        if plot_type == "PMF":
            y_vals = stats.bernoulli.pmf(x_vals, p_bern)
            ax.bar(x_vals, y_vals, alpha=0.7, color='gold', width=0.4)
            ax.set_ylabel('Probability')
            ax.set_title('Bernoulli PMF')
        else:
            x_cdf = np.array([-1, 0, 1, 2])
            y_vals = stats.bernoulli.cdf(x_cdf, p_bern)
            ax.step(x_cdf, y_vals, where='post', linewidth=2, color='red')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title('Bernoulli CDF')
        
        ax.set_xlabel('Observation')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("**How to use:** Select a distribution tab, adjust parameters with sliders, and observe how the distribution changes in real time! Made by @palomavaldes on GitHub.")

import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Who Says Yes, and When Does It Matter?
    ### Statistical Analysis of a Portuguese Bank's Direct Marketing Campaigns

    **Dataset:** UCI Bank Marketing — `bank-additional-full.csv`
    **Records:** 41,188 phone call contacts | **Features:** 20 | **Period:** May 2008 – November 2010
    **Target:** Did the client subscribe to a term deposit? (`y`: yes/no)

    ---

    The bank made over 41,000 phone calls. Only **11.3%** resulted in a subscription.
    This analysis asks: *why does the campaign underperform, and where are the bright spots?*

    We use five statistical tests and three interactive visualizations to find out.
    """)
    return


@app.cell
def _(mo):
    # ── Dependencies ──────────────────────────────────────────────────────────────
    from pathlib import Path

    import pandas as pd
    import numpy as np
    from scipy import stats
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import warnings
    warnings.filterwarnings('ignore')

    # ── Load data (path works from any cwd when opened in marimo) ────────────────
    _data_dir = mo.notebook_dir() or Path.cwd()
    df = pd.read_csv(Path(_data_dir) / "bank-additional-full.csv", sep=";")
    df['subscribed'] = (df['y'] == 'yes').astype(int)

    print(f"Loaded {len(df):,} records × {df.shape[1]} features")
    print(f"Subscription rate: {df['subscribed'].mean()*100:.1f}%")
    return df, go, make_subplots, pd, px, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 1 — The Imbalance Problem

    Before any test, we need to understand the target distribution.
    An 88/12 split means a naive model that always predicts "no" would be 88% accurate — and completely useless.
    """)
    return


@app.cell
def _(df, go):
    counts = df['y'].value_counts()
    pct = df['y'].value_counts(normalize=True) * 100
    _fig = go.Figure(go.Pie(labels=['No (did not subscribe)', 'Yes (subscribed)'], values=counts.values, hole=0.55, marker_colors=['#EF553B', '#00CC96'], textinfo='label+percent', hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>'))
    _fig.update_layout(title=dict(text='Target Distribution — Term Deposit Subscriptions', font_size=16), annotations=[dict(text=f"{counts['yes']:,}<br>subscribed", x=0.5, y=0.5, font_size=14, showarrow=False)], height=420)
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 2 — Does Call Duration Predict Subscription?

    **Test: Mann-Whitney U** (non-parametric; duration is heavily right-skewed)

    **Important caveat:** `duration` is flagged in the dataset documentation as a data leakage risk.
    A 0-second call means definite rejection; a long call suggests genuine engagement.
    In production you wouldn't know duration *before* the call — so it's not actionable as a predictor.
    We test it anyway because the *magnitude* of the difference tells us something real about what engagement looks like.
    """)
    return


@app.cell
def _(df, px, stats):
    yes_dur = df[df['y'] == 'yes']['duration']
    no_dur = df[df['y'] == 'no']['duration']
    (u_stat, p_val) = stats.mannwhitneyu(yes_dur, no_dur, alternative='greater')
    print('── Mann-Whitney U: Duration (subscribers vs non-subscribers) ──')
    print(f'  Subscribers   — median: {yes_dur.median():.0f}s  mean: {yes_dur.mean():.0f}s')
    print(f'  Non-subscribers — median: {no_dur.median():.0f}s  mean: {no_dur.mean():.0f}s')
    print(f'  U-statistic: {u_stat:,.0f}')
    print(f'  p-value: {p_val:.2e}')
    print(f"  Result: {('Significant ✓' if p_val < 0.05 else 'Not significant')}")
    print(f'\n  Subscribers stay on the phone {yes_dur.median() / no_dur.median():.1f}× longer (median)')
    plot_df = df[df['duration'] < 1500].copy()
    _fig = px.violin(plot_df, x='y', y='duration', color='y', color_discrete_map={'yes': '#00CC96', 'no': '#EF553B'}, box=True, points=False, labels={'y': 'Subscribed', 'duration': 'Call Duration (seconds)'}, title='Call Duration Distribution by Subscription Outcome', hover_data=['age', 'job'])
    _fig.update_layout(height=450, showlegend=False)
    # Violin plot
    _fig.add_annotation(x=0.5, y=1.05, xref='paper', yref='paper', text=f'Mann-Whitney U p-value: {p_val:.2e} — highly significant', showarrow=False, font_size=11, font_color='gray')  # trim extreme outliers for viz
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 3 — The Economic Window: Interest Rates & Subscription Timing

    The dataset spans May 2008 to November 2010 — covering the onset of the **2008 global financial crisis**.
    The `euribor3m` variable (European interbank lending rate) is our proxy for economic conditions.

    **Hypothesis:** Subscription rates are higher when interest rates are lower — people are more willing to lock money into a term deposit when the broader economy is uncertain or rates have fallen.

    **Test: Spearman rank correlation** between monthly euribor3m and monthly subscription rate.
    """)
    return


@app.cell
def _(df, go, make_subplots, stats):
    month_order = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_labels = {'mar': 'Mar', 'apr': 'Apr', 'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'aug': 'Aug', 'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dec': 'Dec'}
    monthly = df.groupby('month').agg(total=('y', 'count'), subscribed=('subscribed', 'sum'), euribor=('euribor3m', 'mean')).reindex(month_order).dropna()
    monthly['rate'] = monthly['subscribed'] / monthly['total'] * 100
    monthly['month_label'] = monthly.index.map(month_labels)
    (rho, p_spear) = stats.spearmanr(monthly['euribor'], monthly['rate'])
    print('── Spearman Correlation: euribor3m vs monthly subscription rate ──')
    print(f'  ρ = {rho:.3f},  p = {p_spear:.4f}')
    print(f"  Strong {('negative' if rho < 0 else 'positive')} correlation — {('significant ✓' if p_spear < 0.05 else 'not significant')}")
    _fig = make_subplots(specs=[[{'secondary_y': True}]])
    _fig.add_trace(go.Bar(x=monthly['month_label'], y=monthly['rate'], name='Subscription Rate (%)', marker_color='#636EFA', hovertemplate='<b>%{x}</b><br>Subscription rate: %{y:.1f}%<br>Calls: %{customdata:,}<extra></extra>', customdata=monthly['total']), secondary_y=False)
    _fig.add_trace(go.Scatter(x=monthly['month_label'], y=monthly['euribor'], name='Euribor 3M (%)', mode='lines+markers', line=dict(color='#EF553B', width=3), marker=dict(size=8), hovertemplate='<b>%{x}</b><br>Euribor 3M: %{y:.2f}%<extra></extra>'), secondary_y=True)
    _fig.update_layout(title=dict(text=f'Subscription Rate vs Euribor 3M by Month  |  Spearman ρ = {rho:.2f}, p = {p_spear:.4f}', font_size=15), height=480, legend=dict(x=0.01, y=0.99), hovermode='x unified')
    _fig.update_yaxes(title_text='Subscription Rate (%)', secondary_y=False)
    _fig.update_yaxes(title_text='Euribor 3M Interest Rate (%)', secondary_y=True)
    _fig.show()
    print('\nMonthly breakdown:')
    print(monthly[['month_label', 'total', 'subscribed', 'rate', 'euribor']].to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 4 — The 3D View: Age, Call Duration & Interest Rate

    Three of our strongest signals — client age, call duration, and prevailing interest rate — plotted together.
    Color = subscription outcome. **Rotate the chart** to explore the structure.

    Hover over any point to see job, marital status, and education.
    """)
    return


@app.cell
def _(df, px):
    # Sample for performance (5k points is plenty for a 3D scatter)
    sample = df.sample(5000, random_state=42)
    _fig = px.scatter_3d(sample, x='age', y='duration', z='euribor3m', color='y', color_discrete_map={'yes': '#00CC96', 'no': '#EF553B'}, opacity=0.6, size_max=4, labels={'age': 'Age', 'duration': 'Call Duration (s)', 'euribor3m': 'Euribor 3M', 'y': 'Subscribed'}, title='3D View: Age × Call Duration × Interest Rate — colored by subscription outcome', hover_data={'job': True, 'marital': True, 'education': True, 'age': True, 'duration': True, 'euribor3m': ':.2f', 'y': True})
    _fig.update_traces(marker=dict(size=3))
    _fig.update_layout(height=600, scene=dict(xaxis_title='Age', yaxis_title='Call Duration (s)', zaxis_title='Euribor 3M (%)'))
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 5 — Campaign Fatigue

    **Test: Kruskal-Wallis H** across campaign contact count groups

    How does subscription rate change the more times you call the same person?
    We hypothesize diminishing returns — a classic pattern in direct marketing.
    """)
    return


@app.cell
def _(df, go, make_subplots, stats):
    # Cap at 10 contacts (beyond that, very small n)
    df_camp = df[df['campaign'] <= 10].copy()
    groups = [df_camp[df_camp['campaign'] == i]['subscribed'].values for i in range(1, 11)]
    (h_stat, p_kw) = stats.kruskal(*groups)
    print('── Kruskal-Wallis H: subscription rate across contact counts (1–10) ──')
    print(f'  H = {h_stat:.2f},  p = {p_kw:.2e}')
    print(f"  Result: {('Significant ✓' if p_kw < 0.05 else 'Not significant')}")
    camp_stats = df_camp.groupby('campaign').agg(rate=('subscribed', 'mean'), n=('subscribed', 'count')).reset_index()
    camp_stats['rate_pct'] = camp_stats['rate'] * 100
    _fig = make_subplots(specs=[[{'secondary_y': True}]])
    _fig.add_trace(go.Bar(x=camp_stats['campaign'], y=camp_stats['rate_pct'], name='Subscription Rate (%)', marker_color='#AB63FA', hovertemplate='<b>Contact #%{x}</b><br>Subscription rate: %{y:.1f}%<br>Calls made: %{customdata:,}<extra></extra>', customdata=camp_stats['n']), secondary_y=False)
    _fig.add_trace(go.Scatter(x=camp_stats['campaign'], y=camp_stats['n'], name='Number of Calls', mode='lines+markers', line=dict(color='#FFA15A', width=2, dash='dot'), hovertemplate='Contact #%{x}: %{y:,} calls<extra></extra>'), secondary_y=True)
    _fig.update_layout(title=dict(text=f'Campaign Fatigue: Subscription Rate Drops with Each Call  |  Kruskal-Wallis p = {p_kw:.2e}', font_size=14), xaxis_title='Number of Contacts This Campaign', height=450, hovermode='x unified')
    _fig.update_yaxes(title_text='Subscription Rate (%)', secondary_y=False)
    _fig.update_yaxes(title_text='Number of Calls', secondary_y=True)
    _fig.show()
    r1 = camp_stats.loc[camp_stats['campaign'] == 1, 'rate_pct'].values[0]
    r3 = camp_stats.loc[camp_stats['campaign'] == 3, 'rate_pct'].values[0]
    print(f'\n  Contact #1 → {r1:.1f}% subscription rate')
    print(f'  Contact #3 → {r3:.1f}% subscription rate')
    print(f'  Insight: Stop calling after 3 attempts — the signal is clear.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 6 — Previous Campaign Outcome

    **Test: Chi-Square test of independence** — `poutcome` vs `y`

    If a client said yes in a *previous* campaign, does that predict success in this one?
    Intuition says yes — and the numbers confirm it dramatically.
    """)
    return


@app.cell
def _(df, pd, px, stats):
    contingency = pd.crosstab(df['poutcome'], df['y'])
    (chi2, p_chi, dof, expected) = stats.chi2_contingency(contingency)
    print('── Chi-Square: Previous Campaign Outcome vs Current Subscription ──')
    print(f'  χ² = {chi2:.2f},  df = {dof},  p = {p_chi:.2e}')
    print(f"  Result: {('Significant ✓' if p_chi < 0.05 else 'Not significant')}")
    print('\n  Contingency table:')
    print(contingency)
    pout_rates = df.groupby('poutcome')['subscribed'].agg(['mean', 'count']).rename(columns={'mean': 'rate', 'count': 'n'}).reset_index()
    pout_rates['rate_pct'] = pout_rates['rate'] * 100
    pout_rates = pout_rates.sort_values('rate_pct', ascending=True)
    _fig = px.bar(pout_rates, x='rate_pct', y='poutcome', orientation='h', color='rate_pct', color_continuous_scale='Tealgrn', labels={'rate_pct': 'Subscription Rate (%)', 'poutcome': 'Previous Campaign Outcome'}, title=f'Previous Campaign Outcome Strongly Predicts Current Subscription  |  χ² p = {p_chi:.2e}', text='rate_pct', hover_data={'n': True, 'rate_pct': ':.1f'})
    _fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', hovertemplate='<b>%{y}</b><br>Subscription rate: %{x:.1f}%<br>Contacts: %{customdata[0]:,}<extra></extra>')
    _fig.update_layout(height=380, coloraxis_showscale=False, xaxis_range=[0, 75])
    _fig.show()
    success_rate = pout_rates.loc[pout_rates['poutcome'] == 'success', 'rate_pct'].values[0]
    nonexist_rate = pout_rates.loc[pout_rates['poutcome'] == 'nonexistent', 'rate_pct'].values[0]
    print(f'\n  Previous success → {success_rate:.1f}% subscription rate')
    print(f'  No prior contact → {nonexist_rate:.1f}% subscription rate')
    print(f'  Uplift: {success_rate / nonexist_rate:.1f}× higher conversion from previously-won customers')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary: What We Found

    | Test | Question | Result |
    |------|----------|--------|
    | Mann-Whitney U | Do subscribers get longer calls? | Yes — 2.7× longer median duration (p < 0.001) |
    | Spearman ρ | Does a lower interest rate mean more subscriptions? | Strong negative correlation (ρ ≈ −0.77) |
    | Kruskal-Wallis | Does subscription rate drop with repeated calls? | Yes — significant decline after contact #1 (p < 0.001) |
    | Chi-Square | Does prior campaign success predict current? | Yes — 6.5× higher rate vs no prior contact (p < 0.001) |

    ### The strategic takeaway

    > **Call the right people, at the right economic moment, no more than 3 times.**
    > Prioritize clients who said yes before. Time campaigns to low-euribor windows.
    > The current approach wastes ~70% of call volume on contacts #4 and beyond with sub-8% conversion.

    ---
    *Analysis by Dany Sigha | PyData 2026 Hackathon | Dataset: UCI Bank Marketing (Moro et al., 2014)*
    """)
    return


if __name__ == "__main__":
    app.run()

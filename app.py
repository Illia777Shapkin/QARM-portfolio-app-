# app.py
import streamlit as st
import pandas as pd

from engine import (
    PortfolioConfig,
    load_all_data,
    run_backtest,
    run_today_optimization,
)

def validate_constraints(
    sector_constraints: dict | None,
    esg_constraints: dict | None,
    asset_class_constraints: dict | None,
):
    """
    Basic feasibility checks for constraints after user input.

    - For each group (sector / ESG / asset class):
        - sum of mins <= 1
        - min <= max for each item (if both exist)
    """
    errors: list[str] = []

    # ----- Sector constraints -----
    if sector_constraints:
        total_min = 0.0
        for name, cons in sector_constraints.items():
            mmin = cons.get("min", 0.0)
            mmax = cons.get("max", 1.0)

            if "min" in cons and "max" in cons and mmin > mmax:
                errors.append(
                    f"Sector '{name}': minimum share ({mmin:.2f}) is greater than maximum ({mmax:.2f})."
                )

            total_min += mmin

        if total_min > 1.0 + 1e-8:
            errors.append(
                f"Sum of **minimum sector shares** ({total_min:.2f}) exceeds 100% of the equity slice."
            )

    # ----- ESG constraints -----
    if esg_constraints:
        total_min = 0.0
        for label, cons in esg_constraints.items():
            mmin = cons.get("min", 0.0)
            mmax = cons.get("max", 1.0)

            if "min" in cons and "max" in cons and mmin > mmax:
                errors.append(
                    f"ESG '{label} score': minimum share ({mmin:.2f}) is greater than maximum ({mmax:.2f})."
                )

            total_min += mmin

        if total_min > 1.0 + 1e-8:
            errors.append(
                f"Sum of **minimum ESG shares** ({total_min:.2f}) exceeds 100% of the equity slice."
            )

    # ----- Asset-class constraints -----
    if asset_class_constraints:
        total_min = 0.0
        for ac_name, cons in asset_class_constraints.items():
            mmin = cons.get("min", 0.0)
            mmax = cons.get("max", 1.0)

            if "min" in cons and "max" in cons and mmin > mmax:
                errors.append(
                    f"Asset class '{ac_name}': minimum weight ({mmin:.2f}) is greater than maximum ({mmax:.2f})."
                )

            total_min += mmin

        if total_min > 1.0 + 1e-8:
            errors.append(
                f"Sum of **minimum asset-class weights** ({total_min:.2f}) exceeds 100% of the portfolio."
            )

    return errors



# --------------- GLOBAL DATA (cached) ---------------
@st.cache_data
def get_data():
    return load_all_data()


def main():
    st.set_page_config(
        page_title="QARM Portfolio Manager",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ("About us", "Portfolio optimization"),
    )

    data = get_data()

    if page == "About us":
        page_about()
    elif page == "Portfolio optimization":
        page_portfolio_optimization(data)


# --------------- PAGE 1: ABOUT US ---------------
def page_about():
    st.title("Our Investment Firm")

    st.markdown(
        """
        ### Who we are
        We are a quantitative asset & risk management boutique.

        Our mission is to build **transparent, rule-based portfolios**
        tailored to each client's risk profile, constraints and ESG preferences.

        ### What this app does
        - Builds a diversified multi-asset portfolio (Equity, Fixed Income, Commodities, Alternatives)
        - Applies **sector** and **ESG** constraints inside the equity bucket
        - Applies **asset-class** constraints at the total portfolio level
        - Optimizes using a **Markowitz mean–variance** model with a robust covariance estimator (Ledoit–Wolf)
        - Backtests the strategy over the selected horizon

        Use the *Portfolio optimization* page from the sidebar to try it.
        """
    )


# --------------- PAGE 2: PORTFOLIO OPTIMIZATION ---------------
def page_portfolio_optimization(data):
    st.title("Portfolio Optimization")

    st.markdown(
        """
        This tool builds a **constrained multi-asset portfolio** based on your preferences:

        1. Choose the **market universe & technical settings**  
        2. Refine the **investment universe with filters** (sectors, ESG, asset classes)  
        3. Answer a short **risk profile questionnaire**  
        4. Set **portfolio constraints** (sectors, ESG, asset classes, max weights)  
        5. Run the **optimization & backtest** and analyze the results  
        """
    )

    # ============================================================
    # STEP 1 – GENERAL SETTINGS
    # ============================================================
    st.markdown("### 🧩 Step 1 – General Settings")

    colA, colB, colC, colD = st.columns(4)

    with colA:
        universe_choice = st.radio(
            "Equity Universe",
            options=["SP500", "MSCI"],
            format_func=lambda x: "S&P 500" if x == "SP500" else "MSCI World",
        )

    with colB:
        investment_horizon_years = st.selectbox(
            "Investment Horizon",
            options=[1, 2, 3, 5, 7, 10],
            index=0,
            format_func=lambda x: f"{x} year" if x == 1 else f"{x} years",
        )

    with colC:
        rebalance_label = st.selectbox(
            "Rebalancing Frequency",
            options=["Yearly", "Quarterly", "Monthly"],
            index=0,
        )
        if rebalance_label == "Yearly":
            rebalancing = 12
        elif rebalance_label == "Quarterly":
            rebalancing = 3
        else:
            rebalancing = 1

    with colD:
        est_months = st.selectbox(
            "Estimation Window",
            options=[6, 12, 24, 36, 60],
            index=1,
            format_func=lambda m: f"{m} months",
        )

    st.markdown("---")

    # ============================================================
    # STEP 2 – UNIVERSE & FILTERS
    # ============================================================
    st.markdown("### 🎯 Step 2 – Universe & Filters")

    # --- Get metadata for chosen equity universe and other assets ---
    if universe_choice == "SP500":
        metadata_equity = data["metadata"]["SP500"]
    else:
        metadata_equity = data["metadata"]["MSCI"]

    metadata_other = data["metadata"]["Other"]

    # ---------- 2.1 Equity filters: sectors & ESG ----------
    st.subheader("Equity Filters")

    sectors_available = (
        metadata_equity["SECTOR"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    col_sect, col_esg = st.columns(2)

    with col_sect:
        selected_sectors = st.multiselect(
            "Sectors to include in equity universe",
            options=sectors_available,
            default=sectors_available,
            help="If you select all sectors, no sector filter is applied.",
        )

        if len(selected_sectors) == len(sectors_available) or len(selected_sectors) == 0:
            keep_sectors = None
        else:
            keep_sectors = selected_sectors

    with col_esg:
        esg_options = ["L", "M", "H"]
        selected_esg = st.multiselect(
            "ESG categories to include",
            options=esg_options,
            default=esg_options,
            help="L = Low, M = Medium, H = High. Selecting all applies no ESG filter.",
        )

        if len(selected_esg) == len(esg_options) or len(selected_esg) == 0:
            keep_esg = None
        else:
            keep_esg = selected_esg

    # ---------- 2.2 Other asset classes & instruments ----------
    st.subheader("Other Asset Classes")

    asset_classes_all = (
        metadata_other["ASSET_CLASS"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    selected_asset_classes_other = st.multiselect(
        "Asset classes to include in the universe (beyond equity)",
        options=asset_classes_all,
        default=asset_classes_all,
        help="These asset classes will be available to the optimizer. "
             "Constraints later control how much can be allocated to each.",
    )

    if len(selected_asset_classes_other) == 0:
        selected_asset_classes_other = asset_classes_all.copy()

    keep_ids_by_class = {}

    for ac in selected_asset_classes_other:
        subset = metadata_other[metadata_other["ASSET_CLASS"] == ac]
        ids_in_class = subset.index.astype(str).tolist()

        # Label map for nicer display
        label_map = {}
        if "TICKER" in subset.columns:
            for idx, row in subset.iterrows():
                label_map[str(idx)] = f"{row['TICKER']}"
        elif "NAME" in subset.columns:
            for idx, row in subset.iterrows():
                label_map[str(idx)] = f"{row['NAME']}"
        else:
            for idx in subset.index:
                label_map[str(idx)] = str(idx)

        labeled_options = [f"{id_} – {label_map[id_]}" for id_ in ids_in_class]

        st.markdown(f"**{ac} instruments to include**")
        selected_labels = st.multiselect(
            f"Select {ac} instruments (leave all selected to keep full class)",
            options=labeled_options,
            default=labeled_options,
        )

        selected_ids = [s.split(" – ")[0] for s in selected_labels]

        if 0 < len(selected_ids) < len(ids_in_class):
            keep_ids_by_class[ac] = selected_ids

    keep_ids_by_class = keep_ids_by_class if keep_ids_by_class else None

    st.markdown("---")

    # ============================================================
    # STEP 3 – RISK PROFILE QUESTIONNAIRE → GAMMA
    # ============================================================
    st.markdown("### 📊 Step 3 – Risk Profile Questionnaire")

    st.caption(
        "Answer each question on a 1–5 scale. "
        "1 = very conservative, 5 = very aggressive."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        q1 = st.slider(
            "1. Reaction to a -20% loss in one year\n"
            "1 = sell everything, 5 = buy more",
            min_value=1, max_value=5, value=3,
        )

        q2 = st.slider(
            "2. Comfort with large fluctuations\n"
            "1 = not at all, 5 = very comfortable",
            min_value=1, max_value=5, value=3,
        )

        q3 = st.slider(
            "3. Return vs risk trade-off\n"
            "1 = stable low returns, 5 = max return even with large risk",
            min_value=1, max_value=5, value=3,
        )

        q4 = st.slider(
            "4. Investment horizon\n"
            "1 = < 1 year, 5 = > 10 years",
            min_value=1, max_value=5, value=3,
        )

        q5 = st.slider(
            "5. How do you view risk?\n"
            "1 = something to avoid, 5 = essential for higher returns",
            min_value=1, max_value=5, value=3,
        )

    with col_right:
        q6 = st.slider(
            "6. Stress during market crashes\n"
            "1 = extremely stressed, 5 = not stressed at all",
            min_value=1, max_value=5, value=3,
        )

        q7 = st.slider(
            "7. Stability of your income/finances\n"
            "1 = very unstable, 5 = very stable",
            min_value=1, max_value=5, value=3,
        )

        q8 = st.slider(
            "8. Experience with investing\n"
            "1 = not familiar, 5 = very experienced",
            min_value=1, max_value=5, value=3,
        )

        q9 = st.slider(
            "9. Reaction to a +20% gain in one year\n"
            "1 = sell to lock gains, 5 = add significantly more money",
            min_value=1, max_value=5, value=3,
        )

        q10 = st.slider(
            "10. Share of net worth in risky assets\n"
            "1 = < 10%, 5 = > 60%",
            min_value=1, max_value=5, value=3,
        )

    scores = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
    S = sum(scores)
    gamma = 0.5 + 0.15 * (S - 10)  # internal only

    if S <= 20:
        profile_label = "Very Conservative"
        profile_text = (
            "You have a **very low tolerance for risk** and prefer capital preservation. "
            "The portfolio will be tilted towards safer, lower-volatility assets."
        )
    elif S <= 30:
        profile_label = "Conservative"
        profile_text = (
            "You are **cautious with risk**, but willing to accept some fluctuations. "
            "The portfolio will prioritize stability with a moderate growth component."
        )
    elif S <= 35:
        profile_label = "Balanced"
        profile_text = (
            "You have a **balanced attitude** towards risk and return. "
            "The portfolio will mix growth assets with stabilizing components."
        )
    elif S <= 42:
        profile_label = "Dynamic"
        profile_text = (
            "You are **comfortable with risk** and seek higher returns. "
            "The portfolio will have a strong allocation to growth and risky assets."
        )
    else:
        profile_label = "Aggressive"
        profile_text = (
            "You have a **high risk tolerance** and focus on return maximization. "
            "The portfolio will be heavily exposed to volatile, return-seeking assets."
        )

    st.markdown("")
    col_score, col_profile = st.columns(2)
    with col_score:
        st.metric("Total Risk Score (S)", f"{S} / 50")
    with col_profile:
        st.markdown(f"**Risk Profile:** {profile_label}")
        st.caption(profile_text)

    st.markdown("---")

    # ============================================================
    # STEP 4 – CONSTRAINTS
    # ============================================================
    st.markdown("### 🧱 Step 4 – Constraints")

    st.caption(
        "All constraints are expressed as **fractions** (0.10 = 10%). "
        "Leave min = 0 and max = 1 to avoid imposing a constraint."
    )

    # ------------------------------------------------------------
    # 4.1 Max weight per asset (with safe default + warning)
    # ------------------------------------------------------------
    st.subheader("Maximum Weight per Asset")

    use_custom_max = st.checkbox(
        "Enable custom maximum weight per asset",
        value=False,
        help="By default, each asset is capped at 5%. Enable only if you understand concentration risk."
    )

    if not use_custom_max:
        max_weight_per_asset = 0.05
        st.info("Using default limit: **5% maximum per individual asset**.")
    else:
        max_weight_per_asset = st.slider(
            "Select maximum weight per asset",
            min_value=0.01,
            max_value=0.25,
            value=0.05,
            step=0.01,
            help="Higher caps increase concentration risk and may reduce diversification."
        )

        st.warning(
            "**Caution:** Increasing the maximum weight per asset may significantly raise your "
            "**idiosyncratic risk** and reduce the portfolio's **diversification benefits**. "
            "Large individual exposures can amplify the impact of adverse movements in a single "
            "security, especially during periods of market stress."
        )

    st.markdown("---")

    # ------------------------------------------------------------
    # 4.2 Sector constraints within equity (relative to equity)
    # ------------------------------------------------------------
    st.subheader("Equity Sector Constraints (relative to the equity exposure)")

    if keep_sectors is None:
        sectors_for_constraints = sectors_available
    else:
        sectors_for_constraints = keep_sectors

    sector_constraints = {}
    sector_min_budget = 0.0  # sum of mins so far, must stay <= 1

    for sec in sectors_for_constraints:
        remaining_min_budget = max(0.0, 1.0 - sector_min_budget)

        with st.expander(f"{sec}", expanded=False):
            col_min, col_max = st.columns(2)

            with col_min:
                sec_min = st.number_input(
                    f"Min share of equity in {sec}",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    key=f"sec_min_{sec}",
                )

            # update budget after this min
            sector_min_budget += sec_min

            with col_max:
                sec_max = st.number_input(
                    f"Max share of equity in {sec}",
                    min_value=0.0,  # ensures min <= max
                    max_value=1.0,
                    value=1.0,
                    step=0.01,
                    format="%.2f",
                    key=f"sec_max_{sec}",
                )

            # Professional-style warnings at boundaries
            eps = 1e-8
            if remaining_min_budget > 0 and abs(sec_min - remaining_min_budget) < eps:
                st.warning(
                    f"The minimum allocation entered for **{sec}** is at the upper feasible bound. "
                    "Any higher minimum would force the sum of sector minima above **100% of the equity slice** "
                    "and is therefore not admissible."
                )

            if sec_min > 0 and abs(sec_max - sec_min) < eps:
                st.info(
                    f"For **{sec}**, the minimum and maximum allocations are effectively identical. "
                    "This leaves no flexibility for the optimizer to rebalance within this sector."
                )

        cons = {}
        if sec_min > 0:
            cons["min"] = float(sec_min)
        if sec_max < 1.0:
            cons["max"] = float(sec_max)
        if cons:
            sector_constraints[sec] = cons

    if not sector_constraints:
        sector_constraints = None

    st.markdown("---")

    # ------------------------------------------------------------
    # 4.3 ESG constraints within equity (relative to equity)
    # ------------------------------------------------------------
    st.subheader("Equity ESG Score Constraints (relative to the equity exposure)")

    esg_all_labels = ["L", "M", "H"]
    if keep_esg is None:
        esg_for_constraints = esg_all_labels
    else:
        esg_for_constraints = keep_esg

    esg_constraints = {}
    esg_min_budget = 0.0

    for label in esg_for_constraints:
        remaining_min_budget = max(0.0, 1.0 - esg_min_budget)

        with st.expander(f"ESG {label}", expanded=False):
            col_min, col_max = st.columns(2)

            with col_min:
                esg_min = st.number_input(
                    f"Min share of equity in ESG {label}",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    key=f"esg_min_{label}",
                )

            esg_min_budget += esg_min

            with col_max:
                esg_max = st.number_input(
                    f"Max share of equity in ESG {label}",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.01,
                    format="%.2f",
                    key=f"esg_max_{label}",
                )

            eps = 1e-8
            if remaining_min_budget > 0 and abs(esg_min - remaining_min_budget) < eps:
                st.warning(
                    f"The minimum allocation entered for **ESG {label}** is at the upper feasible bound. "
                    "Any higher minimum would force the sum of ESG minima above **100% of the equity slice** "
                    "and is therefore not admissible."
                )

            if esg_min > 0 and abs(esg_max - esg_min) < eps:
                st.info(
                    f"For **ESG {label}**, the minimum and maximum allocations are effectively identical. "
                    "This leaves no flexibility for the optimizer within this ESG bucket."
                )

        cons = {}
        if esg_min > 0:
            cons["min"] = float(esg_min)
        if esg_max < 1.0:
            cons["max"] = float(esg_max)
        if cons:
            esg_constraints[label] = cons

    if not esg_constraints:
        esg_constraints = None

    st.markdown("---")

    # ------------------------------------------------------------
    # 4.4 Asset-class constraints (total portfolio)
    # ------------------------------------------------------------
    st.subheader("Asset-Class Constraints (total portfolio)")

    asset_classes_for_constraints = ["Equity"] + selected_asset_classes_other

    asset_class_constraints = {}
    ac_min_budget = 0.0

    for ac in asset_classes_for_constraints:
        remaining_min_budget = max(0.0, 1.0 - ac_min_budget)

        with st.expander(f"{ac}", expanded=False):
            col_min, col_max = st.columns(2)

            with col_min:
                ac_min = st.number_input(
                    f"Min portfolio weight in {ac}",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    format="%.2f",
                    key=f"ac_min_{ac}",
                )

            ac_min_budget += ac_min

            with col_max:
                ac_max = st.number_input(
                    f"Max portfolio weight in {ac}",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.05,
                    format="%.2f",
                    key=f"ac_max_{ac}",
                )

            eps = 1e-8
            if remaining_min_budget > 0 and abs(ac_min - remaining_min_budget) < eps:
                st.warning(
                    f"The minimum allocation entered for **{ac}** is at the upper feasible bound. "
                    "Any higher minimum would force the sum of asset-class minima above **100% of the portfolio** "
                    "and is therefore not admissible."
                )

            if ac_min > 0 and abs(ac_max - ac_min) < eps:
                st.info(
                    f"For **{ac}**, the minimum and maximum allocations are effectively identical. "
                    "This leaves no flexibility for the optimizer to reallocate across asset classes."
                )

        cons = {}
        if ac_min > 0:
            cons["min"] = float(ac_min)
        if ac_max < 1.0:
            cons["max"] = float(ac_max)
        if cons:
            asset_class_constraints[ac] = cons

    if not asset_class_constraints:
        asset_class_constraints = None

    st.markdown("---")

    constraint_errors = validate_constraints(
        sector_constraints=sector_constraints,
        esg_constraints=esg_constraints,
        asset_class_constraints=asset_class_constraints,
    )

    if constraint_errors:
        st.error("The current constraint configuration is not feasible:")
        for msg in constraint_errors:
            st.write(f"• {msg}")

    # ============================================================
    # STEP 5 – RUN OPTIMIZATION & BACKTEST
    # ============================================================
    st.markdown("### 🚀 Step 5 – Run Optimization & Backtest")

    run_clicked = st.button(
        "Run Optimization & Backtest",
        type="primary",
        disabled=bool(constraint_errors),
    )

    if run_clicked:
        # 1) Check constraints first
        if constraint_errors:
            st.error("The current constraint configuration is not feasible:")
            for msg in constraint_errors:
                st.write(f"• {msg}")
            st.stop()  # do not run the optimizer

        # 2) Build config only if constraints are okay
        config = PortfolioConfig(
            today_date=pd.Timestamp("2025-10-01"),
            investment_horizon_years=investment_horizon_years,
            est_months=est_months,
            rebalancing=rebalancing,
            gamma=gamma,
            universe_choice=universe_choice,
            keep_sectors=keep_sectors,
            keep_esg=keep_esg,
            selected_asset_classes_other=selected_asset_classes_other,
            keep_ids_by_class=keep_ids_by_class,
            max_weight_per_asset=max_weight_per_asset,
            sector_constraints=sector_constraints,
            esg_constraints=esg_constraints,
            asset_class_constraints=asset_class_constraints,
        )

        # 3) Run engine with friendly error handling
        try:
            with st.spinner("Optimizing and backtesting..."):
                perf, summary_df, debug_weights_df = run_backtest(config, data)
                today_res = run_today_optimization(config, data)
        except ValueError as e:
            # This catches "Optimization failed: Positive directional derivative..." etc.
            st.error(
                "The optimizer could not find a feasible portfolio with the current set of "
                "constraints and per-asset limits."
            )
            st.caption(
                "This typically happens when minimum allocations across sectors, ESG buckets or "
                "asset classes are too tight relative to the available universe and the maximum "
                "weight per asset. Please relax some minimum constraints or increase the maximum "
                "weight per asset, then try again."
            )
            # Optional: show the raw technical message for yourself
            # st.text(f"Technical details: {e}")
            st.stop()

        st.success("Optimization completed.")

        tab_backtest, tab_today = st.tabs(["📈 Backtest", "📌 Today's Portfolio"])

        with tab_backtest:
            st.subheader("Backtest Performance")
            if not perf.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Cumulative Growth of 1 Unit**")
                    st.line_chart(perf["Growth"])
                with col2:
                    st.markdown("**Last Observations**")
                    st.dataframe(perf.tail())
                st.markdown("**Rebalancing Summary (Top-3 Weights)**")
                st.dataframe(summary_df)
            else:
                st.warning("No valid backtest window for the selected settings.")

        with tab_today:
            st.subheader("Today's Optimal Portfolio")

            today_df = today_res["weights"]
            top5 = today_res["top5"]
            alloc_by_ac = today_res["alloc_by_asset_class"]
            sector_in_eq = today_res["sector_in_equity"]
            esg_in_eq = today_res["esg_in_equity"]

            st.markdown("**Top 5 Holdings**")
            st.dataframe(top5)

            colA, colB, colC = st.columns(3)
            with colA:
                st.markdown("**By Asset Class**")
                st.bar_chart(alloc_by_ac)
            with colB:
                st.markdown("**Sector Breakdown (Equity)**")
                if not sector_in_eq.empty:
                    st.bar_chart(sector_in_eq)
                else:
                    st.info("No equity allocation.")
            with colC:
                st.markdown("**ESG Breakdown (Equity)**")
                if not esg_in_eq.empty:
                    st.bar_chart(esg_in_eq)
                else:
                    st.info("No equity allocation.")

            with st.expander("Full Portfolio Weights"):
                st.dataframe(today_df)







if __name__ == "__main__":
    main()

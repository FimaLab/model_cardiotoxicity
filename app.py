import io
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px  # оставим, но для графиков используем matplotlib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Метаболиты: AUC", layout="wide")
st.title("АUC по метаболитам (test / control_neg)")

st.markdown("""
Загрузите Excel-файл (.xlsx или .xls) со столбцами **Drug**, **Group**, **Concentration**, 
после которых идут столбцы метаболитов.

Приложение:
1) показывает исходные данные,  
2) считает средние по *(Drug, Group, Concentration)*,  
3) считает отношения **test/control_neg**,  
4) считает **AUC по дозам методом трапеций** (wide-таблица),  
5) показывает AUC-таблицу с колонкой «Кардиотоксичность» (в начале),  
6) считает **|Z|-score AUC относительно некардиотоксичных** (полная таблица для всех препаратов),  
7) формирует **итоговую таблицу** *(Cardiotoxic, Drug, Metabolite, AUC, |Z|)* по выбранному порогу,  
8) графики «доза–отношение» (в expander) на **matplotlib** с управлением отображения.
""")

uploaded = st.file_uploader("Загрузить Excel-файл", type=["xlsx", "xls"])

# -------- Helpers --------

@st.cache_data(show_spinner=False)
def load_excel(file) -> pd.DataFrame:
    return pd.read_excel(file, sheet_name=0)

def detect_metabolite_columns(columns: List[str]) -> List[str]:
    cols = list(columns)
    if "Concentration" not in cols:
        return []
    start_idx = cols.index("Concentration") + 1
    return cols[start_idx:]

def compute_group_means(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    required = ["Drug", "Group", "Concentration"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Нет обязательных колонок: {', '.join(missing)}")

    metabolite_cols = detect_metabolite_columns(df.columns)
    if not metabolite_cols:
        raise ValueError("Не найдены колонки метаболитов (ожидаются сразу после 'Concentration').")

    df_numeric = df.copy()
    df_numeric["Concentration"] = pd.to_numeric(df_numeric["Concentration"], errors="coerce")
    for c in metabolite_cols:
        df_numeric[c] = pd.to_numeric(df_numeric[c], errors="coerce")

    grouped = (
        df_numeric
        .groupby(["Drug", "Group", "Concentration"], dropna=False, sort=False)[metabolite_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    ordered_cols = ["Drug", "Group", "Concentration"] + metabolite_cols
    return grouped[ordered_cols], metabolite_cols

def ratios_test_vs_zero_control(agg: pd.DataFrame, meta_cols: List[str]) -> pd.DataFrame:
    """Для каждого Drug: test (dose!=0) / control_neg (dose=0)."""
    base = agg[(agg["Group"] == "control_neg") & (agg["Concentration"] == 0)].copy()
    base = base[["Drug"] + meta_cols].rename(columns={c: f"__base__{c}" for c in meta_cols})

    test = agg[(agg["Group"] == "test") & (agg["Concentration"] != 0)].copy()
    merged = test.merge(base, on="Drug", how="left")

    for c in meta_cols:
        denom = merged[f"__base__{c}"].replace({0: pd.NA})
        merged[c] = merged[c] / denom
        merged[c] = merged[c].astype(float)

    merged = merged.drop(columns=[f"__base__{c}" for c in meta_cols])
    merged["Group"] = "test"
    return merged[["Drug", "Group", "Concentration"] + meta_cols]

def trapz_auc(x: np.ndarray, y: np.ndarray) -> float:
    """AUC трапеций: сортируем по x, убираем NaN/inf и дубли доз."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return np.nan
    idx = np.argsort(x, kind="stable")
    x, y = x[idx], y[idx]
    _, unique_idx = np.unique(x, return_index=True)
    x, y = x[unique_idx], y[unique_idx]
    if x.size < 2:
        return np.nan
    return float(np.trapz(y, x))

def compute_auc_wide(df_ratio: pd.DataFrame, meta_cols: List[str]) -> pd.DataFrame:
    """Wide AUC: строка = Drug, колонки = метаболиты (в исходном порядке)."""
    rows = []
    for drug, sub in df_ratio.groupby("Drug", sort=False):
        x = sub["Concentration"].to_numpy(dtype=float)
        auc_row = {"Drug": drug}
        for m in meta_cols:
            y = sub[m].to_numpy(dtype=float)
            auc_row[m] = trapz_auc(x, y)
        rows.append(auc_row)
    auc_wide = pd.DataFrame(rows)
    return auc_wide[["Drug"] + meta_cols]

# ---------- Z-scores vs non-toxic ----------
def zscores_vs_nontoxic(auc_labeled: pd.DataFrame, meta_cols: List[str]) -> pd.DataFrame:
    """
    |Z|-score для ВСЕХ препаратов относительно распределения некардиотоксичных:
    Z = |(x - mean_non_toxic) / std_non_toxic| (ddof=1).
    """
    df_all = auc_labeled[["Drug"] + meta_cols].copy()
    df_non = auc_labeled[auc_labeled["Cardiotoxic"] == False][meta_cols].copy()

    if df_non.empty:
        out = df_all.copy()
        for m in meta_cols:
            out[m] = np.nan
        return out

    mu = df_non.mean(axis=0)
    sigma = df_non.std(axis=0, ddof=1).replace(0, np.nan)

    Z = (df_all[meta_cols] - mu) / sigma
    Z = Z.abs().replace([np.inf, -np.inf], np.nan)

    return pd.concat([auc_labeled[["Drug"]].reset_index(drop=True),
                      Z.reset_index(drop=True)], axis=1)[["Drug"] + meta_cols]

def melt_significant(df_z: pd.DataFrame, meta_cols: List[str], threshold: float) -> pd.DataFrame:
    """Возвращает (Drug, Metabolite, |Z|) только для значений >= threshold."""
    if df_z.empty or threshold <= 0:
        return pd.DataFrame(columns=["Drug", "Metabolite", "|Z|"])
    long = df_z.melt(id_vars=["Drug"], value_vars=meta_cols,
                     var_name="Metabolite", value_name="|Z|")
    return long[np.isfinite(long["|Z|"]) & (long["|Z|"] >= threshold)]

def to_excel_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    """Записать DataFrame в Excel (с сохранением булевых как True/False)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if "Cardiotoxic" in df.columns:
            df = df.copy()
            df["Cardiotoxic"] = df["Cardiotoxic"].astype(bool)
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.read()

# -------------------- APP --------------------
if uploaded is not None:
    try:
        df_raw = load_excel(uploaded)

        # --- Исходные данные ---
        st.subheader("Исходные данные")
        st.dataframe(df_raw, use_container_width=True)
        st.download_button(
            "Скачать исходные данные (Excel)",
            data=to_excel_bytes(df_raw, sheet_name="Raw"),
            file_name="raw.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # --- Средние ---
        df_agg, meta_cols = compute_group_means(df_raw)
        st.subheader("Средние по (Drug, Group, Concentration)")
        st.dataframe(df_agg, use_container_width=True)
        st.download_button(
            "Скачать средние (Excel)",
            data=to_excel_bytes(df_agg, sheet_name="Averages"),
            file_name="averages.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # --- Отношения ---
        st.subheader("Отношения: test/control_neg")
        df_ratio = ratios_test_vs_zero_control(df_agg, meta_cols)
        st.dataframe(df_ratio, use_container_width=True)
        st.download_button(
            "Скачать отношения (Excel)",
            data=to_excel_bytes(df_ratio, sheet_name="Ratios"),
            file_name="ratios.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # --- AUC (wide) + разметка ---
        st.subheader("AUC по метаболитам (метод трапеций) — с разметкой")
        df_auc_wide = compute_auc_wide(df_ratio, meta_cols)

        # состояние чекбоксов
        if "cardiotox_map" not in st.session_state:
            st.session_state.cardiotox_map = {drug: False for drug in df_auc_wide["Drug"]}
        for drug in df_auc_wide["Drug"]:
            st.session_state.cardiotox_map.setdefault(drug, False)

        df_auc_marked = df_auc_wide.copy()
        df_auc_marked["Cardiotoxic"] = df_auc_marked["Drug"].map(st.session_state.cardiotox_map).fillna(False)
        df_auc_marked = df_auc_marked[["Cardiotoxic", "Drug"] + meta_cols]  # чекбокс первым

        edited_auc = st.data_editor(
            df_auc_marked,
            use_container_width=True,
            key="auc_editor",
            column_config={
                "Drug": st.column_config.TextColumn("Drug", disabled=True),
                "Cardiotoxic": st.column_config.CheckboxColumn(
                    label="Кардиотоксичность",
                    help="Отметьте, если препарат кардиотоксичен",
                    default=False,
                ),
            },
            disabled=meta_cols,
        )
        # sync state
        for _, row in edited_auc.iterrows():
            st.session_state.cardiotox_map[row["Drug"]] = bool(row["Cardiotoxic"])

        st.download_button(
            "Скачать AUC (с разметкой)",
            data=to_excel_bytes(edited_auc, sheet_name="AUC_Wide_Labeled"),
            file_name="auc_wide_with_labels.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # списки групп (для информации)
        st.markdown("### Разметка препаратов")
        col_a, col_b = st.columns(2)
        with col_a:
            tox_list = [d for d, v in st.session_state.cardiotox_map.items() if v]
            st.markdown("**Кардиотоксичные:**")
            st.markdown("\n".join(f"- {d}" for d in tox_list) if tox_list else "_нет отмеченных_")
        with col_b:
            non_list = [d for d, v in st.session_state.cardiotox_map.items() if not v]
            st.markdown("**Без кардиотоксичности:**")
            st.markdown("\n".join(f"- {d}" for d in non_list) if non_list else "_все отмечены как кардиотоксичные_")

        # --- Условие: нужно >= 3 кардиотоксичных для Z и финальной таблицы ---
        n_toxic = len(tox_list)
        can_compute_z = n_toxic >= 3

        if can_compute_z:
            # --- Полная таблица Z относительно некардиотоксичных ---
            st.subheader("|Z|-score AUC относительно некардиотоксичных (полная таблица)")
            auc_for_z = pd.concat(
                [edited_auc[["Drug"]], edited_auc.drop(columns=["Cardiotoxic", "Drug"])],
                axis=1
            )
            z_all = zscores_vs_nontoxic(auc_for_z.assign(Cardiotoxic=edited_auc["Cardiotoxic"]), meta_cols)
            st.dataframe(z_all, use_container_width=True)
            st.download_button(
                "Скачать Z (полная)",
                data=to_excel_bytes(z_all, sheet_name="Z_All_vs_NonToxic"),
                file_name="z_all_vs_nontoxic.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # --- Порог и итоговая таблица ---
            st.subheader("Итоговая таблица (Cardiotoxic, Drug, Metabolite, AUC, |Z|)")
            th_label_global = st.selectbox("Порог по |Z| для итоговой таблицы",
                                           ["|Z| ≥ 1.0", "|Z| ≥ 1.5", "|Z| ≥ 2.0", "|Z| ≥ 3.0"], index=0)
            threshold_global = float(th_label_global.split("≥")[1].strip())

            # значимые пары (Drug, Metabolite, |Z|)
            significant_global = melt_significant(z_all, meta_cols, threshold_global)

            # long AUC с разметкой
            auc_long = edited_auc.melt(
                id_vars=["Cardiotoxic", "Drug"], value_vars=meta_cols,
                var_name="Metabolite", value_name="AUC"
            )

            # объединяем с |Z|
            final = auc_long.merge(significant_global, on=["Drug", "Metabolite"], how="inner")[
                ["Cardiotoxic", "Drug", "Metabolite", "AUC", "|Z|"]
            ]
            st.dataframe(final, use_container_width=True)
            st.download_button(
                f"Скачать итоговую таблицу ({th_label_global})",
                data=to_excel_bytes(final, sheet_name="Final"),
                file_name="final_significant_auc.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.warning("Для расчёта Z-score и итоговой таблицы отметьте **минимум 3 кардиотоксичных** препарата.")
            z_all = None  # для логики фильтра метаболитов в графиках

        # --- Графики (в expander, matplotlib) ---
        with st.expander("Графики доза–отношение (matplotlib)", expanded=False):
            # Глобальные опции графиков
            fill_area = st.checkbox("Закрашивать площадь (AUC)", value=True)

            # Фильтры по кардиотоксичности для графиков (если есть хоть один токсичный)
            show_group_filter = len(tox_list) > 0 or len(non_list) > 0
            group_choice = "Все"
            if show_group_filter:
                group_choice = st.selectbox("Показывать препараты", ["Все", "Только кардиотоксичные", "Только некардиотоксичные"], index=0)

            # ---- ГЛАВЕНСТВУЮЩАЯ фильтрация по порогу |Z| ----
            meta_options = meta_cols
            drugs_allowed_by_z = None  # None = без ограничений по порогу
            if can_compute_z and z_all is not None:
                filter_metas_by_z = st.checkbox("Фильтровать метаболиты и препараты по порогу |Z| (для графиков)", value=False)
                if filter_metas_by_z:
                    sig_graph = melt_significant(z_all, meta_cols, threshold_global)
                    significant_metas_set = set(sig_graph["Metabolite"].unique().tolist())
                    drugs_allowed_by_z = set(sig_graph["Drug"].unique().tolist())

                    meta_options = [m for m in meta_cols if m in significant_metas_set]

                    if not meta_options or not drugs_allowed_by_z:
                        st.info(f"По порогу {th_label_global} не найдено совпадений для графиков.")
                        st.stop()

            c1, c2 = st.columns([1, 2])
            with c1:
                selected_metabolite = st.selectbox("Метаболит для графиков", meta_options, index=0)
            with c2:
                drugs_all = list(pd.unique(df_ratio["Drug"].dropna()))

                if isinstance(drugs_allowed_by_z, set):
                    drugs_all = [d for d in drugs_all if d in drugs_allowed_by_z]

                if group_choice != "Все":
                    tox_map = st.session_state.cardiotox_map
                    if group_choice == "Только кардиотоксичные":
                        drugs_all = [d for d in drugs_all if tox_map.get(d, False)]
                    elif group_choice == "Только некардиотоксичные":
                        drugs_all = [d for d in drugs_all if not tox_map.get(d, False)]

                if not drugs_all:
                    st.info("По выбранным фильтрам (включая порог |Z|) не найдено совпадений для графиков.")
                    st.stop()

                selected_drugs = st.multiselect("Препараты", options=drugs_all, default=drugs_all)

            plot_df = df_ratio[df_ratio["Drug"].isin(selected_drugs)] if selected_drugs else df_ratio.iloc[0:0]

            if plot_df.empty:
                st.info("По выбранным фильтрам (включая порог |Z|) не найдено совпадений для графиков.")
                st.stop()

            # Рисуем по одному графику на препарат (matplotlib)
            for drug, sub in plot_df.groupby("Drug", sort=False):
                sub_sorted = sub.sort_values("Concentration")
                x = sub_sorted["Concentration"].to_numpy(dtype=float)
                y = sub_sorted[selected_metabolite].to_numpy(dtype=float)

                st.markdown(f"### {drug}")

                # Индивидуальные подписи и заголовок для каждого графика
                # Две колонки: слева настройки, справа сам график
                col_settings, col_plot = st.columns([1, 2])

                with col_settings:
                    xlabel = st.text_input(
                        f"Подпись оси X — {drug}",
                        value="Concentration",
                        help="Эта подпись применяется для всех графиков данного препарата.",
                        key=f"xlabel_{drug}"
                    )
                    ylabel = st.text_input(
                        f"Подпись оси Y — {selected_metabolite} ({drug})",
                        value=selected_metabolite,
                        help="Эта подпись относится только к текущему метаболиту в группе выбранного препарата.",
                        key=f"ylabel_{drug}_{selected_metabolite}"
                    )
                    default_title = f"{selected_metabolite} — {drug}"
                    title_text = st.text_input(
                        "Заголовок графика",
                        value=default_title,
                        key=f"title_{drug}_{selected_metabolite}"
                    )

                with col_plot:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(x, y, marker="o")
                    if fill_area:
                        ax.fill_between(x, y, 0, alpha=0.2)

                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    ax.set_title(title_text)
                    # сетку не показываем

                    st.pyplot(fig, use_container_width=True)

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=600, bbox_inches="tight")
                    buf.seek(0)
                    st.download_button(
                        label="Скачать график (PNG, 600 dpi)",
                        data=buf.getvalue(),
                        file_name=f"{drug}_{selected_metabolite}.png",
                        mime="image/png",
                        key=f"download_{drug}_{selected_metabolite}"
                    )
                    plt.close(fig)

    except Exception as ex:
        st.error(f"Ошибка при чтении файла: {ex}")
else:
    st.info("Загрузите Excel-файл, чтобы увидеть данные и расчёты.")

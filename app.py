import io
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Метаболиты: AUC", layout="wide")
st.title("AUC по метаболитам (test / control_neg)")

st.markdown("""
Загрузите Excel-файл (.xlsx или .xls) со столбцами **Drug**, **Group**, **Concentration**, 
после которых идут столбцы метаболитов.

Приложение:
1) показывает исходные данные,  
2) считает средние по *(Drug, Group, Concentration)*,  
3) считает отношения **test/control_neg**,  
4) считает **AUC по дозам методом трапеций** (wide-таблица),  
5) показывает AUC-таблицу с колонкой «Кардиотоксичность»,  
6) считает **|Z|-score AUC** отдельно для кардио/некардио и фильтрует **метаболиты** по порогам (|Z| ≥ 1, 1.5, 2, 3),  
7) графики «доза–отношение» (в expander).
""")

uploaded = st.file_uploader("Загрузить Excel-файл", type=["xlsx", "xls"])

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

    # сохраняем порядок появления ключей (sort=False), без дополнительной сортировки
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
        for m in meta_cols:  # исходный порядок метаболитов
            y = sub[m].to_numpy(dtype=float)
            auc_row[m] = trapz_auc(x, y)
        rows.append(auc_row)
    auc_wide = pd.DataFrame(rows)
    return auc_wide[["Drug"] + meta_cols]

def zscore_by_group_abs(auc_wide_labeled: pd.DataFrame, meta_cols: List[str], group_flag: bool) -> pd.DataFrame:
    """
    |Z|-score AUC для указанной группы (Cardiotoxic=True/False).
    Z = |(x - mean_group)/std_group|, ddof=1. Порядок колонок сохраняется как в meta_cols.
    """
    df = auc_wide_labeled[auc_wide_labeled["Cardiotoxic"] == group_flag].copy()
    if df.empty:
        return pd.DataFrame(columns=["Drug"] + meta_cols)
    Z = df[meta_cols].apply(lambda s: np.abs((s - s.mean()) / s.std(ddof=1)), axis=0)
    out = pd.concat([df[["Drug"]].reset_index(drop=True), Z.reset_index(drop=True)], axis=1)
    return out[["Drug"] + meta_cols]

def filter_metabolites_by_threshold(df_zabs: pd.DataFrame, meta_cols: List[str], threshold: float) -> pd.DataFrame:
    """
    Оставляет только те метаболиты (колонки), у которых хотя бы у одного препарата |Z| >= threshold.
    Строки (Drug) не фильтруем.
    """
    if df_zabs.empty:
        return df_zabs
    keep = ["Drug"] + [m for m in meta_cols if (df_zabs[m] >= threshold).any(skipna=True)]
    # если ни один метаболит не прошёл порог — оставим только Drug
    if keep == ["Drug"]:
        return df_zabs[["Drug"]]
    return df_zabs[keep]

def melt_hits(df_zabs: pd.DataFrame, meta_cols: List[str], threshold: float) -> pd.DataFrame:
    """Long-вид хитов: Drug, Metabolite, |Z| для значений >= threshold."""
    if df_zabs.empty or threshold <= 0:
        return pd.DataFrame(columns=["Drug", "Metabolite", "|Z|"])
    long = df_zabs.melt(id_vars=["Drug"], value_vars=meta_cols, var_name="Metabolite", value_name="|Z|")
    hits = long[np.isfinite(long["|Z|"]) & (long["|Z|"] >= threshold)]
    return hits

def to_excel_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.read()

if uploaded is not None:
    try:
        df_raw = load_excel(uploaded)

        st.subheader("Исходные данные")
        st.dataframe(df_raw, use_container_width=True)

        try:
            df_agg, meta_cols = compute_group_means(df_raw)
        except ValueError as e:
            st.error(str(e))
        else:
            st.subheader("Средние по (Drug, Group, Concentration)")
            st.dataframe(df_agg, use_container_width=True)

            st.subheader("Отношения: test/control_neg")
            df_ratio = ratios_test_vs_zero_control(df_agg, meta_cols)
            st.dataframe(df_ratio, use_container_width=True)

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
                disabled=[c for c in df_auc_marked.columns if c not in ["Drug", "Cardiotoxic"]],
            )

            # обновить state из редактора
            for _, row in edited_auc.iterrows():
                st.session_state.cardiotox_map[row["Drug"]] = bool(row["Cardiotoxic"])

            # списки групп
            toxic_list = [d for d, v in st.session_state.cardiotox_map.items() if v]
            nontoxic_list = [d for d, v in st.session_state.cardiotox_map.items() if not v]

            st.markdown("### Разметка препаратов")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Кардиотоксичные:**")
                st.markdown("\n".join(f"- {d}" for d in toxic_list) if toxic_list else "_нет отмеченных_")
            with col_b:
                st.markdown("**Без кардиотоксичности:**")
                st.markdown("\n".join(f"- {d}" for d in nontoxic_list) if nontoxic_list else "_все отмечены как кардиотоксичные_")

            # --- |Z|-SCORE AUC ПО ГРУППАМ + ФИЛЬТР МЕТАБОЛИТОВ ---
            st.subheader("|Z|-score AUC по группам и пороги (фильтруем метаболиты)")
            auc_labeled = edited_auc.copy()

            zabs_toxic = zscore_by_group_abs(auc_labeled, meta_cols, group_flag=True)
            zabs_nontoxic = zscore_by_group_abs(auc_labeled, meta_cols, group_flag=False)

            # селектор порога (всегда используем >= threshold)
            th_label = st.selectbox(
                "Порог по |Z| (метаболит включается, если у какого-либо препарата |Z| ≥ порога)",
                ["Показать все", "|Z| ≥ 1.0", "|Z| ≥ 1.5", "|Z| ≥ 2.0", "|Z| ≥ 3.0"],
                index=0
            )
            if th_label == "Показать все":
                threshold = 0.0
            else:
                threshold = float(th_label.split("≥")[1].strip())

            # отфильтруем МЕТАБОЛИТЫ (колонки), а не препараты (строки)
            zabs_toxic_f = filter_metabolites_by_threshold(zabs_toxic, meta_cols, threshold) if threshold > 0 else zabs_toxic
            zabs_nontoxic_f = filter_metabolites_by_threshold(zabs_nontoxic, meta_cols, threshold) if threshold > 0 else zabs_nontoxic

            czt, czn = st.columns(2)
            with czt:
                st.markdown("**Кардиотоксичные — |Z|-score**")
                st.dataframe(zabs_toxic_f, use_container_width=True)
            with czn:
                st.markdown("**Без кардиотоксичности — |Z|-score**")
                st.dataframe(zabs_nontoxic_f, use_container_width=True)

            # хиты (long) по выбранному порогу (полезно для экспорта/анализа)
            st.markdown("### Хиты (Drug, Metabolite, |Z|) по выбранному порогу")
            hits_toxic = melt_hits(zabs_toxic, meta_cols, threshold)
            hits_nontoxic = melt_hits(zabs_nontoxic, meta_cols, threshold)

            cht, chn = st.columns(2)
            with cht:
                st.markdown(f"**Кардиотоксичные — хиты ({th_label})**")
                st.dataframe(hits_toxic, use_container_width=True)
            with chn:
                st.markdown(f"**Без кардиотоксичности — хиты ({th_label})**")
                st.dataframe(hits_nontoxic, use_container_width=True)

            # --- Кнопки скачивания ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="Скачать AUC (с разметкой) — Excel",
                    data=to_excel_bytes(auc_labeled, sheet_name="AUC_Wide_Labeled"),
                    file_name="auc_wide_with_labels.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with col2:
                # выгрузка |Z|-таблиц (полных) двумя листами
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    zabs_toxic.to_excel(writer, index=False, sheet_name="Zabs_Toxic")
                    zabs_nontoxic.to_excel(writer, index=False, sheet_name="Zabs_NonToxic")
                buffer.seek(0)
                st.download_button(
                    label="Скачать |Z|-таблицы (полные)",
                    data=buffer.read(),
                    file_name="zabs_tables_full.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with col3:
                # выгрузка отфильтрованных по метаболитам таблиц
                buffer2 = io.BytesIO()
                with pd.ExcelWriter(buffer2, engine="openpyxl") as writer:
                    zabs_toxic_f.to_excel(writer, index=False, sheet_name="Zabs_Toxic_Filtered")
                    zabs_nontoxic_f.to_excel(writer, index=False, sheet_name="Zabs_NonToxic_Filtered")
                    hits_toxic.to_excel(writer, index=False, sheet_name="Hits_Toxic")
                    hits_nontoxic.to_excel(writer, index=False, sheet_name="Hits_NonToxic")
                buffer2.seek(0)
                st.download_button(
                    label=f"Скачать отфильтрованные |Z| (порог {th_label})",
                    data=buffer2.read(),
                    file_name="zabs_filtered_and_hits.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            # --- Графики в expander ---
            with st.expander("Графики доза–отношение", expanded=False):
                c1, c2 = st.columns([1, 2])
                with c1:
                    selected_metabolite = st.selectbox("Метаболит", meta_cols)
                with c2:
                    # порядок препаратов — как встречаются в данных
                    drugs_all = list(pd.unique(df_ratio["Drug"].dropna()))
                    selected_drugs = st.multiselect(
                        "Препараты (Drug)",
                        options=drugs_all,
                        default=drugs_all
                    )

                plot_df = df_ratio[df_ratio["Drug"].isin(selected_drugs)] if selected_drugs else df_ratio.iloc[0:0]

                for drug, sub in plot_df.groupby("Drug", sort=False):
                    st.markdown(f"### {drug}")
                    fig = px.line(
                        sub.sort_values("Concentration"),
                        x="Concentration",
                        y=selected_metabolite,
                        markers=True,
                        title=f"{selected_metabolite} — {drug}",
                    )
                    fig.update_traces(fill="tozeroy", fillcolor="rgba(0,100,200,0.2)")
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as ex:
        st.error(f"Ошибка при чтении файла: {ex}")
else:
    st.info("Загрузите Excel-файл, чтобы увидеть данные и расчёты.")

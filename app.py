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
3) считает отношения **test/control_neg** и **control_neg/control_neg (точка 0)**,  
4) считает **AUC по дозам методом трапеций относительно baseline=1** (wide-таблица, по модулю или знаковая — на выбор),  
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
    """
    Для каждого Drug:
      - baseline: control_neg/control_neg (строка при Concentration=0, все метаболиты=1)
      - test(dose!=0)/control_neg(0)
    Вывод — в логичном порядке: для каждого препарата сначала baseline=0, затем его test-дозы.
    """
    result_rows = []

    for drug, sub in agg.groupby("Drug", sort=False):
        base = sub[(sub["Group"] == "control_neg") & (sub["Concentration"] == 0)].copy()
        if base.empty:
            continue

        # baseline строка (все = 1)
        baseline = {"Drug": drug, "Group": "control_neg", "Concentration": 0.0}
        for c in meta_cols:
            baseline[c] = 1.0
        result_rows.append(baseline)

        # test строки (dose != 0)
        test = sub[(sub["Group"] == "test") & (sub["Concentration"] != 0)].copy()
        for _, row in test.iterrows():
            row_out = {"Drug": drug, "Group": "test", "Concentration": row["Concentration"]}
            for c in meta_cols:
                denom = base[c].iloc[0]
                row_out[c] = row[c] / denom if pd.notna(denom) and denom != 0 else np.nan
            result_rows.append(row_out)

    return pd.DataFrame(result_rows, columns=["Drug", "Group", "Concentration"] + meta_cols)

def auc_relative_to_baseline(x: np.ndarray, y: np.ndarray, baseline: float = 1.0, use_abs: bool = True) -> float:
    """
    AUC относительно baseline.
    Если use_abs=True — считаем по модулю (всё >= 0).
    Если use_abs=False — знаковая площадь (выше baseline полож., ниже — отрицат.).
    Корректно обрабатывает пересечения с baseline (линейная интерполяция).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return np.nan

    idx = np.argsort(x, kind="stable")
    x, y = x[idx], y[idx]

    # удаляем дубли по x
    _, unique_idx = np.unique(x, return_index=True)
    x, y = x[unique_idx], y[unique_idx]
    if x.size < 2:
        return np.nan

    total = 0.0
    b = float(baseline)

    for i in range(len(x) - 1):
        x1, x2 = float(x[i]), float(x[i+1])
        y1, y2 = float(y[i]), float(y[i+1])
        d1, d2 = y1 - b, y2 - b
        dx = x2 - x1
        if not np.isfinite(d1) or not np.isfinite(d2) or dx <= 0:
            continue

        same_side = (d1 >= 0 and d2 >= 0) or (d1 <= 0 and d2 <= 0)
        if same_side:
            a1, a2 = (abs(d1), abs(d2)) if use_abs else (d1, d2)
            total += 0.5 * (a1 + a2) * dx
        else:
            # пересечение baseline: точка по линейной интерполяции
            t = abs(d1) / (abs(d1) + abs(d2))  # доля от x1 до пересечения
            x_cross = x1 + t * dx
            dx1, dx2 = x_cross - x1, x2 - x_cross
            if use_abs:
                area1 = 0.5 * (abs(d1) + 0.0) * dx1
                area2 = 0.5 * (0.0 + abs(d2)) * dx2
            else:
                area1 = 0.5 * (d1 + 0.0) * dx1
                area2 = 0.5 * (0.0 + d2) * dx2
            total += area1 + area2

    return float(total)

def compute_auc_wide(df_ratio: pd.DataFrame, meta_cols: List[str], use_abs: bool) -> pd.DataFrame:
    """
    Wide AUC: строка = Drug, колонки = метаболиты (в исходном порядке).
    Используется AUC относительно baseline=1 (по модулю или знаковая — в зависимости от use_abs).
    """
    rows = []
    for drug, sub in df_ratio.groupby("Drug", sort=False):
        x = sub["Concentration"].to_numpy(dtype=float)
        auc_row = {"Drug": drug}
        for m in meta_cols:
            y = sub[m].to_numpy(dtype=float)
            auc_row[m] = auc_relative_to_baseline(x, y, baseline=1.0, use_abs=use_abs)
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
        st.subheader("Отношения: test/control_neg (и baseline 0: control_neg/control_neg)")
        df_ratio = ratios_test_vs_zero_control(df_agg, meta_cols)
        st.dataframe(df_ratio, use_container_width=True)
        st.download_button(
            "Скачать отношения (Excel)",
            data=to_excel_bytes(df_ratio, sheet_name="Ratios"),
            file_name="ratios.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # === Переключатель способа интегрирования AUC (дОЛЖЕН быть выше таблицы AUC) ===
        use_abs_auc = st.checkbox(
            "Считать AUC по **модулю** относительно baseline=1",
            value=True,
            help="Если включено — площадь выше и ниже baseline суммируется как положительная. "
                 "Если выключено — учитывается знак (ниже baseline — отрицательная).",
        )

        # --- AUC (wide) + разметка ---
        st.subheader("AUC по метаболитам (метод трапеций, относительно baseline=1) — с разметкой")
        df_auc_wide = compute_auc_wide(df_ratio, meta_cols, use_abs=use_abs_auc)

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
            z_all = None  # для логики метаболитов в графиках

        # --- Графики (в expander, matplotlib) ---
        with st.expander("Графики доза–отношение (matplotlib)", expanded=False):
            # Главная настройка графиков
            fill_area = st.checkbox("Закрашивать площадь (AUC)", value=True)

            # ==== 1) ГЛАВЕНСТВУЮЩИЙ фильтр по |Z| (стоит выше остальных виджетов) ====
            filter_by_z = False
            sig_graph = pd.DataFrame(columns=["Drug", "Metabolite", "|Z|"])
            if can_compute_z and z_all is not None:
                filter_by_z = st.checkbox(
                    "Фильтровать по порогу |Z| (распространяется на выбор метаболитов и препаратов)",
                    value=False
                )
                if filter_by_z:
                    sig_graph = melt_significant(z_all, meta_cols, threshold_global)

            # ==== 2) Список метаболитов с учётом главного фильтра ====
            if filter_by_z:
                sig_metas = set(sig_graph["Metabolite"].unique().tolist())
                meta_options = [m for m in meta_cols if m in sig_metas]
                if not meta_options:
                    st.info(f"По порогу {th_label_global} не найдено значимых метаболитов для отображения.")
                    st.stop()
            else:
                meta_options = meta_cols

            selected_metabolite = st.selectbox("Метаболит для графиков", meta_options, index=0)

            # ==== 3) Базовый список препаратов ====
            drugs_all = list(pd.unique(df_ratio["Drug"].dropna()))

            # если включён главный фильтр, то ограничиваем препараты только теми, где выбранный метаболит значим
            if filter_by_z:
                drugs_allowed_for_meta = set(
                    sig_graph.loc[sig_graph["Metabolite"] == selected_metabolite, "Drug"].unique().tolist()
                )
                drugs_all = [d for d in drugs_all if d in drugs_allowed_for_meta]
                if not drugs_all:
                    st.info(f"По порогу {th_label_global} и выбранному метаболиту "
                            f"«{selected_metabolite}» нет подходящих препаратов.")
                    st.stop()

            # ==== 4) Фильтр «Показывать препараты» ниже по приоритету ====
            tox_list = [d for d, v in st.session_state.cardiotox_map.items() if v]
            non_list = [d for d, v in st.session_state.cardiotox_map.items() if not v]
            show_group_filter = len(tox_list) > 0 or len(non_list) > 0
            group_choice = "Все"
            if show_group_filter:
                group_choice = st.selectbox(
                    "Показывать препараты",
                    ["Все", "Только кардиотоксичные", "Только некардиотоксичные"],
                    index=0
                )
                tox_map = st.session_state.cardiotox_map
                if group_choice == "Только кардиотоксичные":
                    drugs_all = [d for d in drugs_all if tox_map.get(d, False)]
                elif group_choice == "Только некардиотоксичные":
                    drugs_all = [d for d in drugs_all if not tox_map.get(d, False)]

            if not drugs_all:
                st.info("По выбранным фильтрам не найдено подходящих препаратов для построения графиков.")
                st.stop()

            selected_drugs = st.multiselect("Препараты", options=drugs_all, default=drugs_all)

            # ==== 5) Данные для графиков ====
            plot_df = df_ratio[df_ratio["Drug"].isin(selected_drugs)] if selected_drugs else df_ratio.iloc[0:0]

            # если включён главный фильтр по Z — оставляем только пары (Drug, selected_metabolite), проходящие порог
            if filter_by_z and not plot_df.empty:
                allowed_pairs = set(
                    tuple(x) for x in sig_graph[["Drug", "Metabolite"]].itertuples(index=False, name=None)
                )
                plot_df = plot_df[
                    (plot_df["Drug"].isin(drugs_all)) &
                    (plot_df["Drug"].map(lambda d: (d, selected_metabolite) in allowed_pairs))
                ]

            if plot_df.empty:
                st.info("По выбранным настройкам (включая порог |Z| и выбранный метаболит) совпадений не найдено.")
            else:
                # ==== 6) Отрисовка: слева настройки, справа график ====
                for drug, sub in plot_df.groupby("Drug", sort=False):
                    sub_sorted = sub.sort_values("Concentration")
                    x = sub_sorted["Concentration"].to_numpy(dtype=float)
                    y = sub_sorted[selected_metabolite].to_numpy(dtype=float)

                    st.markdown(f"### {drug}")

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
                            ax.fill_between(x, y, 1.0, alpha=0.2)  # закраска к baseline=1

                        # горизонтальная линия y=1 (baseline)
                        ax.axhline(1.0, linestyle="--", linewidth=1, alpha=0.8)

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

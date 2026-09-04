"""Streamlit app to browse and cull GONG pretraining images.

Shows images 8 at a time in a 2x4 grid with Prev/Next batch navigation
and a per-image Delete button.

Run with:
    streamlit run scripts/pretrain_data/image_browser.py
"""

import os

import matplotlib.pyplot as plt
import streamlit as st

from image_similarity import METHODS, similarity_matrix

IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "processed",
    "gong_pretrain",
)
BATCH_SIZE = 8
GRID_COLS = 4


@st.cache_data
def list_images(image_dir: str) -> list[str]:
    if not os.path.isdir(image_dir):
        return []
    return sorted(
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    )


@st.cache_data
def compute_similarity(image_dir: str, filenames: tuple[str, ...], method: str):
    paths = [os.path.join(image_dir, f) for f in filenames]
    return similarity_matrix(paths, method=method)


def render_similarity_heatmap(matrix, filenames: list[str]):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    n = len(filenames)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n))
    ax.set_yticklabels([f"{i}: {f[:18]}" for i, f in enumerate(filenames)])
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7
            )
    fig.colorbar(im, ax=ax, label="similarity")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="GONG Pretrain Image Browser", layout="wide")
    st.title("GONG Pretrain Image Browser")

    if "start" not in st.session_state:
        st.session_state.start = 0

    images = list_images(IMAGE_DIR)
    total = len(images)

    if total == 0:
        st.warning(f"No images found in {IMAGE_DIR}")
        return

    # Clamp start in case images were deleted since last render.
    max_start = max(0, ((total - 1) // BATCH_SIZE) * BATCH_SIZE)
    st.session_state.start = min(st.session_state.start, max_start)
    st.session_state.start = max(st.session_state.start, 0)

    def go_prev():
        st.session_state.start = max(0, st.session_state.start - BATCH_SIZE)

    def go_next():
        st.session_state.start = min(max_start, st.session_state.start + BATCH_SIZE)

    def delete_image(filename: str):
        path = os.path.join(IMAGE_DIR, filename)
        if os.path.exists(path):
            os.remove(path)
        list_images.clear()

    start = st.session_state.start
    end = min(start + BATCH_SIZE, total)

    nav_left, nav_mid, nav_right = st.columns([1, 3, 1])
    with nav_left:
        st.button(
            "⬅ Prev", on_click=go_prev, disabled=(start == 0), use_container_width=True
        )
    with nav_mid:
        st.markdown(
            f"<div style='text-align:center'>Images {start + 1}-{end} of {total}</div>",
            unsafe_allow_html=True,
        )
    with nav_right:
        st.button(
            "Next ➡",
            on_click=go_next,
            disabled=(end >= total),
            use_container_width=True,
        )

    batch = images[start:end]

    for row_start in range(0, len(batch), GRID_COLS):
        row = batch[row_start : row_start + GRID_COLS]
        cols = st.columns(GRID_COLS)
        for col, filename in zip(cols, row):
            with col:
                st.image(os.path.join(IMAGE_DIR, filename), use_container_width=True)
                st.caption(filename)
                st.button(
                    "🗑️ Delete",
                    key=f"delete_{filename}",
                    on_click=delete_image,
                    args=(filename,),
                    use_container_width=True,
                )

    st.divider()
    with st.expander("🔍 Pairwise similarity (this batch)", expanded=False):
        method = st.selectbox(
            "Method (classical CV, not deep learning)",
            options=list(METHODS.keys()),
            index=list(METHODS.keys()).index("combined"),
            help=(
                "histogram: grayscale intensity histogram correlation\n"
                "ssim: structural similarity index\n"
                "orb: ORB keypoint feature matching\n"
                "combined: average of histogram + ssim"
            ),
        )
        if st.button("Compute similarity for current batch"):
            if len(batch) < 2:
                st.info("Need at least 2 images in the batch to compare.")
            else:
                matrix = compute_similarity(IMAGE_DIR, tuple(batch), method)
                render_similarity_heatmap(matrix, batch)

    st.divider()
    bottom_left, bottom_mid, bottom_right = st.columns([1, 3, 1])
    with bottom_left:
        st.button(
            "⬅ Prev",
            on_click=go_prev,
            disabled=(start == 0),
            use_container_width=True,
            key="prev_bottom",
        )
    with bottom_right:
        st.button(
            "Next ➡",
            on_click=go_next,
            disabled=(end >= total),
            use_container_width=True,
            key="next_bottom",
        )


if __name__ == "__main__":
    main()

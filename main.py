import os
import uuid
import json
from glob import glob
from typing import Optional, List, Dict, Any
from fastapi import HTTPException
from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fpdf import FPDF
import traceback
import numpy as np
import pandas as pd
import scanpy as sc
import umap
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


# --- File Paths ---
UPLOAD_FOLDER = "uploads"
PLOT_FOLDER = "static/plots"
TEMPLATES_FOLDER = "templates"
REPORTS_FOLDER = "reports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(os.path.join("static", "css"), exist_ok=True)
os.makedirs(os.path.join("static", "js"), exist_ok=True)
os.makedirs(os.path.join("static", "downloads"), exist_ok=True)

# Make scanpy save figures into our static folder
sc.settings.figdir = PLOT_FOLDER

# --- Single FastAPI App Instance ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/reports", StaticFiles(directory="reports"), name="reports") # Mount the reports directory
templates = Jinja2Templates(directory=TEMPLATES_FOLDER)

# Cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary storage for background processing status
processing_status = {}

# ---------------------- Helpers ----------------------


def align_index(df: pd.DataFrame, adata: sc.AnnData) -> pd.DataFrame:
    """
    Align CSV DataFrame index to adata.obs_names safely.
    Handles:
      - Prefix mismatch (e.g. 'BIOKEY_2_Pre_')
      - Order mismatch
      - Missing cells → fill with 0
    """
    obs_idx = pd.Index(adata.obs_names.astype(str))
    df_idx = pd.Index(df.index.astype(str))

    # Case 1: Direct match
    common = df_idx.intersection(obs_idx)
    if not common.empty:
        return df.reindex(obs_idx, fill_value=0)

    # Case 2: Add prefix
    prefixed = "BIOKEY_2_Pre_" + df_idx
    common = prefixed.intersection(obs_idx)
    if not common.empty:
        df.index = prefixed
        return df.reindex(obs_idx, fill_value=0)

    # Case 3: Strip prefix
    stripped = obs_idx.str.replace(r"^BIOKEY_2_Pre_", "", regex=True)
    common = df_idx.intersection(stripped)
    if not common.empty:
        obs_idx = stripped
        return df.reindex(obs_idx, fill_value=0)

    print("⚠️ No strong match found. Returning zeros with CSV columns.")
    return pd.DataFrame(0, index=obs_idx, columns=df.columns)


def _to_dense(a):
    if sparse.issparse(a):
        return a.toarray()
    return np.asarray(a)

def _get_adata_path(file_id: str) -> Optional[str]:
    """
    Returns the path to the processed H5AD file for a given file_id.
    """
    folder = os.path.join(UPLOAD_FOLDER, file_id)
    h5ad_path = os.path.join(folder, "processed.h5ad")
    if os.path.exists(h5ad_path):
        return h5ad_path
    return None

def _load_adata_safe(file_id: str) -> sc.AnnData:
    """
    Safely loads the processed AnnData object.
    Raises FileNotFoundError if the file does not exist.
    """
    file_path = _get_adata_path(file_id)
    if not file_path:
        raise FileNotFoundError(f"Processed file not found for id: {file_id}")
    return sc.read_h5ad(file_path)

def _get_activity_file(file_id: str) -> str:
    """
    Returns the file path for the uploaded activity CSV file.
    Raises FileNotFoundError if not found.
    """
    file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_activity_scores.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Activity file not found for id: {file_id}")
    return file_path

def _load_activity_csv(file_id: str) -> pd.DataFrame:
    file_path = _get_activity_file(file_id)
    print("🔍 Loading CSV from:", file_path)   # Debugging ke liye
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found for id: {file_id}")
    return pd.read_csv(file_path, index_col=0)


def resolve_csv_path(file_id: str) -> Optional[str]:
    """Return the path to the activity_scores.csv file inside the folder."""
    folder = os.path.join(UPLOAD_FOLDER, file_id)
    csv_path = os.path.join(folder, "activity_scores.csv")
    if os.path.exists(csv_path):
        return csv_path
    return None

def run_dimensionality_reduction(adata_obj: sc.AnnData) -> None:
    """
    Performs PCA and UMAP on numeric columns in adata.obs starting with 'H_graph_' or 'H_std_'.
    Handles missing values and stores embeddings in adata.obsm.
    """
    h_graph_cols = [col for col in adata_obj.obs.columns if str(col).startswith('H_graph_')]
    h_std_cols = [col for col in adata_obj.obs.columns if str(col).startswith('H_std_')]
    
    # Process H_graph data
    if h_graph_cols:
        h_graph_data = adata_obj.obs[h_graph_cols].to_numpy(dtype=float)
        if np.isnan(h_graph_data).any():
            h_graph_data = SimpleImputer(strategy='mean').fit_transform(h_graph_data)
        
        if h_graph_data.shape[1] >= 2:
            h_graph_pca = PCA(n_components=min(2, h_graph_data.shape[1]))
            adata_obj.obsm['X_hgraph_pca'] = h_graph_pca.fit_transform(h_graph_data)
        
        if h_graph_data.shape[1] >= 2:
            reducer_graph = umap.UMAP(random_state=42)
            adata_obj.obsm['X_hgraph_umap'] = reducer_graph.fit_transform(h_graph_data)

    # Process H_std data
    if h_std_cols:
        h_std_data = adata_obj.obs[h_std_cols].to_numpy(dtype=float)
        if np.isnan(h_std_data).any():
            h_std_data = SimpleImputer(strategy='mean').fit_transform(h_std_data)
            
        if h_std_data.shape[1] >= 2:
            h_std_pca = PCA(n_components=min(2, h_std_data.shape[1]))
            adata_obj.obsm['X_hstd_pca'] = h_std_pca.fit_transform(h_std_data)

        if h_std_data.shape[1] >= 2:
            reducer_std = umap.UMAP(random_state=42)
            adata_obj.obsm['X_hstd_umap'] = reducer_std.fit_transform(h_std_data)
            
def find_latest_folder_with_files():
    folders = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isdir(os.path.join(UPLOAD_FOLDER, f))]
    if not folders:
        return None
    folders.sort(key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)), reverse=True)
    return folders[0]

        

def plot_embedding(
    adata_obj: sc.AnnData,
    embedding_data: np.ndarray,
    output_path: str,
    title: str,
    color_by_column: str = None, 
    max_points: int = 5000,
    max_categories: int = 10,
    colors: str = "tab20"   # 👈 new optional arg
) -> None:
    """Plots a 2D embedding with specified color column."""
    
    n_points = embedding_data.shape[0]
    
    if color_by_column not in adata_obj.obs.columns:
        color_by_column = None
    
    # Subsample points if too many
    if n_points > max_points:
        idx = np.random.choice(n_points, max_points, replace=False)
        embedding_data = embedding_data[idx, :]
        obs_df = adata_obj.obs.iloc[idx].copy()
    else:
        obs_df = adata_obj.obs.copy()
    
    plt.figure(figsize=(8, 6))
    
    if color_by_column:
        color_data = obs_df[color_by_column]
        if color_data.nunique() > max_categories:
            top_categories = color_data.value_counts().nlargest(max_categories).index
            color_data = color_data.apply(lambda x: x if x in top_categories else 'Other')
        
        sns.scatterplot(
            x=embedding_data[:, 0],
            y=embedding_data[:, 1],
            hue=color_data,
            s=20,
            alpha=0.7,
            palette=colors   # 👈 use colors arg
        )
        plt.legend(title=color_by_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        sns.scatterplot(
            x=embedding_data[:, 0],
            y=embedding_data[:, 1],
            s=20,
            alpha=0.7
        )
    
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _uniq_plot_path(prefix: str) -> tuple[str, str]:
    name = f"{prefix}_{uuid.uuid4().hex}.png"
    return os.path.join(PLOT_FOLDER, name), f"/static/plots/{name}"


def process_full_data(file_id: str):
    """Use precomputed activity_scores CSV directly, no W/H computation."""
    folder = os.path.join(UPLOAD_FOLDER, file_id)
    h5ad_path = os.path.join(folder, "processed.h5ad")
    csv_path = os.path.join(folder, "activity_scores.csv")

    try:
        adata = sc.read_h5ad(h5ad_path)

        if os.path.exists(csv_path):
            act_df = pd.read_csv(csv_path, index_col=0)
            act_df_aligned = align_index(act_df, adata)
            
            # Use concat for a cleaner merge, handling potential duplicates gracefully
            adata.obs = pd.concat([adata.obs, act_df_aligned], axis=1)

        run_dimensionality_reduction(adata)
        adata.write(h5ad_path)
        processing_status[file_id] = "done"

    except Exception as e:
        processing_status[file_id] = f"error: {str(e)}"
        print(f"Error in process_full_data: {e}")
        traceback.print_exc()

def generate_full_report(file_id: str, pdf_path: str, top_n: int = 50, top_cols: list | None = None):
    from fpdf import FPDF
    import os, json
    import pandas as pd
    from glob import glob

    # Landscape mode for wide tables
    pdf = FPDF(orientation='L')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ===== Page 1: Project Summary =====
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Project Full Report: {file_id}", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Project Summary", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"This report contains full results for project ID: {file_id}.")
    pdf.ln(5)

    # ===== Page 2: Plots =====
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Generated Plots", ln=True)
    plot_files = sorted(glob(os.path.join(PLOT_FOLDER, f"*{file_id}*.png")))
    if plot_files:
        for plot_path in plot_files:
            plot_name = os.path.basename(plot_path).replace("_", " ").replace(".png", "").title()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, plot_name, ln=True)
            pdf.image(plot_path, w=180)
            pdf.ln(5)
    else:
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, "No plots generated for this project.", ln=True)

    # ===== Page 3: DE Results =====
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "DE Results", ln=True)
    try:
        adata = _load_adata_safe(file_id)
        if 'up_down_table' in adata.uns:
            de_data = json.loads(adata.uns['up_down_table'])
            if de_data:
                headers = list(de_data[0].keys())
                col_width = min(50, pdf.w / (len(headers)+1))  # dynamic width
                pdf.set_font("Arial", "B", 10)
                for header in headers:
                    pdf.cell(col_width, 10, header, border=1, align="C")
                pdf.ln()
                pdf.set_font("Arial", "", 8)
                for row in de_data[:50]:
                    for header in headers:
                        value = str(row.get(header, ""))
                        if len(value) > 30:
                            value = value[:27] + "..."
                        pdf.cell(col_width, 8, value, border=1)
                    pdf.ln()
                if len(de_data) > 50:
                    pdf.cell(0, 8, f"...({len(de_data)-50} more rows)", ln=True)
            else:
                pdf.cell(0, 8, "No DE results found.", ln=True)
    except Exception as e:
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Error loading DE results: {str(e)}", ln=True)
        print(f"[DEBUG] DE Results error: {e}")

    # ===== Page 4: Merged CSV Table (UI snapshot) =====
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Merged CSV Table (UI view)", ln=True)
    try:
        merged_csv_path = os.path.join(REPORTS_FOLDER, f"{file_id}_merged.csv")

        if os.path.exists(merged_csv_path):
            df_merged = pd.read_csv(merged_csv_path)
        else:
            df_merged = pd.DataFrame()

        if not df_merged.empty:
            headers = list(df_merged.columns)
            col_width = min(50, pdf.w / (len(headers)+1))  # dynamic width
            pdf.set_font("Arial", "B", 10)
            for h in headers:
                pdf.cell(col_width, 10, h, border=1, align="C")
            pdf.ln()
            pdf.set_font("Arial", "", 8)
            for _, row in df_merged.iterrows():
                for h in headers:
                    value = str(row[h])
                    if len(value) > 30:
                        value = value[:27] + "..."
                    pdf.cell(col_width, 8, value, border=1)
                pdf.ln()
        else:
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 8, "No merged CSV snapshot available.", ln=True)

    except Exception as e:
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Error loading merged CSV: {str(e)}", ln=True)
        print(f"[DEBUG] merged CSV error: {e}")

    # ===== Save PDF =====
    dir_path = os.path.dirname(pdf_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    pdf.output(pdf_path)
    print(f"[INFO] Full report generated at: {pdf_path}")




# ---------------------- Routes ----------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    index_path = os.path.join(TEMPLATES_FOLDER, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>Metabolomics Dashboard</h1><p>Frontend file not found. Please ensure index.html is in the 'templates' folder.</p>")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/list_uploads")
async def list_uploads():
    ids = []
    for d in sorted(glob(os.path.join(UPLOAD_FOLDER, "*")), key=os.path.getmtime, reverse=True):
        if os.path.isdir(d):
            ids.append({"file_id": os.path.basename(d), "modified": os.path.getmtime(d)})
    return {"uploads": ids}

@app.post("/upload_h5ad_and_csvs")
async def upload_h5ad_and_csvs(
    background_tasks: BackgroundTasks,
    h5ad_file: UploadFile = File(...),
    w_std_file: Optional[UploadFile] = File(None),
    w_graph_file: Optional[UploadFile] = File(None),
    activity_scores_file: Optional[UploadFile] = File(None),
    metabolic_file: Optional[UploadFile] = File(None)   # 🔥 new param
):
    """
    Upload H5AD + W_std + W_graph + activity_scores + Metabolic CSVs
    """
    file_id = str(uuid.uuid4())
    folder = os.path.join(UPLOAD_FOLDER, file_id)
    os.makedirs(folder, exist_ok=True)

    # Save H5AD file
    h5ad_path = os.path.join(folder, "processed.h5ad")
    try:
        with open(h5ad_path, "wb") as f:
            while chunk := await h5ad_file.read(8192):
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save H5AD file: {str(e)}")

    # Save W_std CSV
    if w_std_file:
        w_std_path = os.path.join(folder, "W_std.csv")
        try:
            with open(w_std_path, "wb") as f:
                while chunk := await w_std_file.read(8192):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save W_std CSV: {str(e)}")

    # Save W_graph CSV
    if w_graph_file:
        w_graph_path = os.path.join(folder, "W_graph.csv")
        try:
            with open(w_graph_path, "wb") as f:
                while chunk := await w_graph_file.read(8192):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save W_graph CSV: {str(e)}")

    # Save activity_scores CSV
    if activity_scores_file:
        activity_scores_path = os.path.join(folder, "activity_scores.csv")
        try:
            with open(activity_scores_path, "wb") as f:
                while chunk := await activity_scores_file.read(8192):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save activity_scores CSV: {str(e)}")

    # Save metabolic CSV  🔥
    if metabolic_file:
        metabolic_path = os.path.join(folder, "metabolic.csv")
        try:
            with open(metabolic_path, "wb") as f:
                while chunk := await metabolic_file.read(8192):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save metabolic CSV: {str(e)}")

    # Start background processing
    processing_status[file_id] = "processing"
    background_tasks.add_task(process_full_data, file_id)

    return {
        "file_id": file_id,
        "status": "processing_started",
        "message": "All files uploaded and processing started in the background."
    }
@app.post("/upload_h5ad_and_csvs")
async def upload_h5ad_and_csvs(
    background_tasks: BackgroundTasks,
    h5ad_file: UploadFile = File(...),
    w_std_file: Optional[UploadFile] = File(None),
    w_graph_file: Optional[UploadFile] = File(None),
    activity_scores_file: Optional[UploadFile] = File(None),
    metabolic_file: Optional[UploadFile] = File(None)   # 🔥 new param
):
    """
    Upload H5AD + W_std + W_graph + activity_scores + Metabolic CSVs
    """
    file_id = str(uuid.uuid4())
    folder = os.path.join(UPLOAD_FOLDER, file_id)
    os.makedirs(folder, exist_ok=True)

    # Save H5AD file
    h5ad_path = os.path.join(folder, "processed.h5ad")
    try:
        with open(h5ad_path, "wb") as f:
            while chunk := await h5ad_file.read(8192):
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save H5AD file: {str(e)}")

    # Save W_std CSV
    if w_std_file:
        w_std_path = os.path.join(folder, "W_std.csv")
        try:
            with open(w_std_path, "wb") as f:
                while chunk := await w_std_file.read(8192):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save W_std CSV: {str(e)}")

    # Save W_graph CSV
    if w_graph_file:
        w_graph_path = os.path.join(folder, "W_graph.csv")
        try:
            with open(w_graph_path, "wb") as f:
                while chunk := await w_graph_file.read(8192):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save W_graph CSV: {str(e)}")

    # Save activity_scores CSV
    if activity_scores_file:
        activity_scores_path = os.path.join(folder, "activity_scores.csv")
        try:
            with open(activity_scores_path, "wb") as f:
                while chunk := await activity_scores_file.read(8192):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save activity_scores CSV: {str(e)}")

    # Save metabolic CSV  🔥
    if metabolic_file:
        metabolic_path = os.path.join(folder, "metabolic.csv")
        try:
            with open(metabolic_path, "wb") as f:
                while chunk := await metabolic_file.read(8192):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save metabolic CSV: {str(e)}")

    # Start background processing
    processing_status[file_id] = "processing"
    background_tasks.add_task(process_full_data, file_id)

    return {
        "file_id": file_id,
        "status": "processing_started",
        "message": "All files uploaded and processing started in the background."
    }
@app.post("/upload_h5ad_and_csvs")
async def upload_h5ad_and_csvs(
    background_tasks: BackgroundTasks,
    h5ad_file: UploadFile = File(...),
    w_std_file: Optional[UploadFile] = File(None),
    w_graph_file: Optional[UploadFile] = File(None),
    activity_scores_file: Optional[UploadFile] = File(None),
    metabolic_file: Optional[UploadFile] = File(None)   # 🔥 new param
):
    """
    Upload H5AD + W_std + W_graph + activity_scores + Metabolic CSVs
    """
    file_id = str(uuid.uuid4())
    folder = os.path.join(UPLOAD_FOLDER, file_id)
    os.makedirs(folder, exist_ok=True)

    # Save H5AD file
    h5ad_path = os.path.join(folder, "processed.h5ad")
    try:
        with open(h5ad_path, "wb") as f:
            while chunk := await h5ad_file.read(8192):
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save H5AD file: {str(e)}")

    # Save W_std CSV
    if w_std_file:
        w_std_path = os.path.join(folder, "W_std.csv")
        try:
            with open(w_std_path, "wb") as f:
                while chunk := await w_std_file.read(8192):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save W_std CSV: {str(e)}")

    # Save W_graph CSV
    if w_graph_file:
        w_graph_path = os.path.join(folder, "W_graph.csv")
        try:
            with open(w_graph_path, "wb") as f:
                while chunk := await w_graph_file.read(8192):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save W_graph CSV: {str(e)}")

    # Save activity_scores CSV
    if activity_scores_file:
        activity_scores_path = os.path.join(folder, "activity_scores.csv")
        try:
            with open(activity_scores_path, "wb") as f:
                while chunk := await activity_scores_file.read(8192):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save activity_scores CSV: {str(e)}")

    # Save metabolic CSV  🔥
    if metabolic_file:
        metabolic_path = os.path.join(folder, "metabolic.csv")
        try:
            with open(metabolic_path, "wb") as f:
                while chunk := await metabolic_file.read(8192):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save metabolic CSV: {str(e)}")

    # Start background processing
    processing_status[file_id] = "processing"
    background_tasks.add_task(process_full_data, file_id)

    return {
        "file_id": file_id,
        "status": "processing_started",
        "message": "All files uploaded and processing started in the background."
    }
@app.get("/check_status/{file_id}")
async def check_status(file_id: str):
    """Check if full processing is done"""
    status = processing_status.get(file_id, "not_found")
    return {
        "file_id": file_id,
        "status": status,
        "message": f"Current status: {status}"
    }



from fastapi import Query, Request, HTTPException

# ----------------- Settings -----------------
FIXED_OBS_COLS = ['patient_id', 'cellType', 'cohort']  # Important columns only
MAX_CELLS_PER_GROUP = 200  # Limit cells per group for speed

def combine_obs_columns(adata: sc.AnnData, cols: list, df_index: pd.Index):
    """Combine multiple obs columns into a short group string."""
    metadata = adata.obs[cols].reindex(df_index).astype(str)
    # Shorten group string by using only first 3 chars per column
    metadata["combined_group"] = metadata.apply(lambda x: "_".join([s[:3] for s in x]), axis=1)
    return metadata


# ----------------- Activity Score Barplot -----------------
@app.get("/generate_activity_score_plot")
@app.post("/generate_activity_score_plot")
async def generate_activity_score_plot(
    request: Request, 
    file_id: str = Query(None), 
    top_n: int = Query(10), 
    tasks: str = Query(None)  # comma-separated tasks from dropdown
):
    if request.method == "POST" and (not file_id or file_id == "null"):
        form = await request.form()
        file_id = form.get("file_id")
        top_n = int(form.get("top_n", 10))
        tasks = form.get("tasks")

    if not file_id:
        raise HTTPException(status_code=400, detail="file_id is required")

    csv_path = resolve_csv_path(file_id)
    h5ad_path = _get_adata_path(file_id)
    if not csv_path or not h5ad_path:
        raise HTTPException(status_code=404, detail="Files not found for this project ID")

    try:
        df = pd.read_csv(csv_path, index_col=0).T
        adata = sc.read_h5ad(h5ad_path)
        df = align_index(df, adata)

        metadata = combine_obs_columns(adata, FIXED_OBS_COLS, df.index)

        # Sample cells per group
        sampled_idx = []
        for grp, sub in df.groupby(metadata["combined_group"]):
            if len(sub) > MAX_CELLS_PER_GROUP:
                sampled_idx.extend(sub.sample(MAX_CELLS_PER_GROUP, random_state=42).index)
            else:
                sampled_idx.extend(sub.index)
        df = df.loc[sampled_idx]
        metadata = metadata.loc[sampled_idx]

        df_clustered = df.groupby(metadata["combined_group"]).mean()

        # Filter selected tasks if provided
        if tasks:
            selected_tasks = [t.strip() for t in tasks.split(",") if t.strip() in df_clustered.columns]
            if not selected_tasks:
                raise HTTPException(status_code=400, detail="No valid tasks selected")
            df_top = df_clustered[selected_tasks]
        else:
            top_activities = df_clustered.var(axis=0).sort_values(ascending=False).head(top_n).index
            df_top = df_clustered[top_activities]

        plt.figure(figsize=(12, 6))
        df_top.T.plot(kind="bar", figsize=(12, 6))
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Activity Score")
        plt.title("Top Activity Scores per Combined Group")
        plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")

        plot_path, url = _uniq_plot_path(f"activity_score_{file_id}")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        return {"activity_score_plot_url": url}

    except Exception as e:
        import traceback
        print("🔥 Activity Score Plot failed:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



# ----------------- Dotplot -----------------
@app.get("/generate_dotplot")
@app.post("/generate_dotplot")
async def generate_dotplot(
    request: Request, 
    file_id: str = Query(None), 
    top_n: int = Query(10), 
    tasks: str = Query(None)  # comma-separated tasks
):
    if request.method == "POST" and (not file_id or file_id == "null"):
        form = await request.form()
        file_id = form.get("file_id")
        top_n = int(form.get("top_n", 10))
        tasks = form.get("tasks")

    if not file_id:
        raise HTTPException(status_code=400, detail="file_id is required")

    csv_path = resolve_csv_path(file_id)
    h5ad_path = _get_adata_path(file_id)
    if not csv_path or not h5ad_path:
        raise HTTPException(status_code=404, detail="Files not found for this project ID")

    try:
        df = pd.read_csv(csv_path, index_col=0).T
        adata = sc.read_h5ad(h5ad_path)
        df = align_index(df, adata)

        metadata = combine_obs_columns(adata, FIXED_OBS_COLS, df.index)

        # Sample cells per group
        sampled_idx = []
        for grp, sub in df.groupby(metadata["combined_group"]):
            if len(sub) > MAX_CELLS_PER_GROUP:
                sampled_idx.extend(sub.sample(MAX_CELLS_PER_GROUP, random_state=42).index)
            else:
                sampled_idx.extend(sub.index)
        df = df.loc[sampled_idx]
        metadata = metadata.loc[sampled_idx]

        df_clustered = df.groupby(metadata["combined_group"])

        # Determine top activities
        if tasks:
            top_activities = [t.strip() for t in tasks.split(",") if t.strip() in df.columns]
            if not top_activities:
                raise HTTPException(status_code=400, detail="No valid tasks selected")
        else:
            top_activities = df_clustered.mean().var(axis=0).sort_values(ascending=False).head(top_n).index

        plot_data = []
        for grp, sub in df_clustered:
            sub_top = sub[top_activities]
            frac = (sub_top > 0).sum() / len(sub_top)
            mean_vals = sub_top.mean()
            for activity in top_activities:
                plot_data.append({
                    "Group": grp,
                    "Activity": activity,
                    "MeanScore": mean_vals[activity],
                    "Fraction": frac[activity]
                })

        plot_df = pd.DataFrame(plot_data)
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            data=plot_df,
            x="Activity",
            y="Group",
            size="Fraction",
            hue="MeanScore",
            palette="viridis",
            sizes=(20, 200),
            legend="brief"
        )
        plt.xticks(rotation=45, ha="right")
        plt.title("Dotplot of Top Activity Scores per Combined Group")

        plot_path, url = _uniq_plot_path(f"dotplot_{file_id}")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        return {"dotplot_url": url}

    except Exception as e:
        import traceback
        print("🔥 Dotplot failed:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- Merge CSVs -----------------
@app.get("/merge_csvs")
async def merge_csvs(top_n: int = Query(40), top_cols: int = Query(None)):
    folder = find_latest_folder_with_files()
    if not folder:
        return JSONResponse(content={"error": "No folder with files found"}, status_code=404)

    folder_path = os.path.join(UPLOAD_FOLDER, folder)
    metabolic_file = os.path.join(folder_path, "metabolic.csv")
    activity_file = os.path.join(folder_path, "activity_scores.csv")

    if not os.path.exists(metabolic_file) or not os.path.exists(activity_file):
        return JSONResponse(content={"error": "Required CSV files not found"}, status_code=404)
 
    try:
        # Read CSVs in chunks for large files
        df_met = pd.concat(pd.read_csv(metabolic_file, chunksize=100000), ignore_index=True)
        df_act = pd.concat(pd.read_csv(activity_file, chunksize=100000), ignore_index=True)

        # Transpose activity values so rows = metabolic pathways
        df_act_t = pd.DataFrame(df_act.values.T)

        # Align rows
        min_rows = min(len(df_met), len(df_act_t))
        df_met = df_met.iloc[:min_rows]
        df_act_t = df_act_t.iloc[:min_rows]

        # Rename activity columns (optional, numeric to meaningful)
        df_act_t.columns = [f"Activity_{i+1}" for i in range(df_act_t.shape[1])]

        # Insert Metabolic column first
        metabolic_col = df_met.iloc[:, 0]  # first column from metabolic file
        df_final = df_act_t.copy()
        df_final.insert(0, "Metabolic", metabolic_col)

        # Optional: limit columns while keeping Metabolic first
        if top_cols:
            cols_ordered = ["Metabolic"] + [c for c in df_final.columns if c != "Metabolic"]
            df_final = df_final[cols_ordered[:top_cols]]

        # Top N rows
        df_final = df_final.head(top_n)

        # Replace NaN/Inf with empty string
        df_final.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        df_final.fillna("", inplace=True)

        # ================== ✅ Save snapshot CSV for PDF ==================
        save_path = os.path.join(REPORTS_FOLDER, f"{folder}_merged.csv")
        df_final.to_csv(save_path, index=False)

        # Convert to JSON for response
        result = df_final.to_dict(orient='records')
        return JSONResponse(content={"merged_data": result, "saved_csv": save_path})

    except Exception as e:
        return JSONResponse(content={"error": f"Error during merge: {str(e)}"}, status_code=500)

    
@app.get("/generate_qc_plots")
async def generate_qc_plots(file_id: str, palette: str = "Categorical"):
    try:
        adata = _load_adata_safe(file_id)
        if adata.obs.shape[0] == 0:
            raise HTTPException(status_code=400, detail="H5AD obs is empty")

        count_col = next((c for c in ["nCount_RNA","total_counts"] if c in adata.obs.columns), None)
        feature_col = next((c for c in ["nFeature_RNA","n_genes_by_counts"] if c in adata.obs.columns), None)
        mito_col = next((c for c in ["percent.mito","pct_counts_mt"] if c in adata.obs.columns), None)

        if not count_col or not feature_col:
            raise HTTPException(status_code=400, detail="Required columns missing in H5AD")

        # --- Use Carbon palette from color_palettes dict ---
        colors = color_palettes.get(palette, color_palettes["Categorical"])

        output_urls = {}

        # --- Scatter plot ---
        scatter_path, scatter_url = _uniq_plot_path(f"qc_scatter_{file_id}")
        plt.figure(figsize=(6,5))
        params = {'x': adata.obs[count_col], 'y': adata.obs[feature_col], 's': 12, 'linewidth': 0, 'alpha': 0.8}
        if mito_col:
            params['hue'] = adata.obs[mito_col]
            params['palette'] = colors   # 👈 Carbon palette
        sns.scatterplot(**params)
        if mito_col:
            plt.legend(title=mito_col, bbox_to_anchor=(1.05,1), loc='upper left')
        plt.xlabel(count_col)
        plt.ylabel(feature_col)
        plt.title("QC Scatterplot")
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close()
        output_urls['scatter_url'] = scatter_url

        # --- Raw histogram ---
        raw_hist_path, raw_hist_url = _uniq_plot_path(f"qc_raw_hist_{file_id}")
        plt.figure(figsize=(6,5))
        sns.histplot(
            adata.obs[count_col], 
            bins=80, 
            kde=False, 
            color=colors[0],  # first color of Carbon palette
            edgecolor="black"
        )
        plt.xlabel(count_col)
        plt.ylabel("Cell count")
        plt.title(f"Distribution of {count_col}")
        plt.tight_layout()
        plt.savefig(raw_hist_path, dpi=150, bbox_inches="tight")
        plt.close()
        output_urls['raw_hist_url'] = raw_hist_url

        # --- Log histogram ---
        log_hist_path, log_hist_url = _uniq_plot_path(f"qc_log_hist_{file_id}")
        plt.figure(figsize=(6,5))
        sns.histplot(
            np.log1p(adata.obs[count_col]), 
            bins=80, 
            kde=False, 
            color=colors[1] if len(colors) > 1 else colors[0],  # second color
            edgecolor="black"
        )
        plt.xlabel(f"log1p({count_col})")
        plt.ylabel("Cell count")
        plt.title(f"Shifted log of {count_col}")
        plt.tight_layout()
        plt.savefig(log_hist_path, dpi=150, bbox_inches="tight")
        plt.close()
        output_urls['log_hist_url'] = log_hist_url

        return output_urls

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"QC plot failed: {str(e)}")





 
from matplotlib.patches import Patch
@app.post("/generate_heatmap")
async def generate_heatmap(
    file_id: str = Form(...),
    num_rows: int = Form(50),
    num_cols: int = Form(50)
):
    csv_path = resolve_csv_path(file_id)
    h5ad_path = _get_adata_path(file_id)

    if not csv_path or not h5ad_path:
        raise HTTPException(status_code=404, detail="Files not found for this project ID")

    try:
        # Load CSV (genes x cells)
        df = pd.read_csv(csv_path, index_col=0).T  # rows=cells, cols=genes

        # Load h5ad
        adata = sc.read_h5ad(h5ad_path)
        anno_cols = [c for c in [
            'Cluster', 'Cell Type', 'patient_id', 'cellType'
        ] if c in adata.obs.columns]
        metadata = adata.obs[anno_cols]

        # Align cells
        common_cells = df.index.intersection(metadata.index)
        if common_cells.empty:
            raise HTTPException(status_code=400, detail="No matching cells between CSV and h5ad.obs")

        df = df.loc[common_cells]
        metadata = metadata.loc[common_cells]

        # Subset
        df_subset = df.iloc[:min(num_rows, df.shape[0]), :min(num_cols, df.shape[1])]
        metadata_subset = metadata.loc[df_subset.index]

        # ---- Color mapping ----
        row_colors_list = []
        lut_maps = {}

        for col in metadata_subset.columns:
            values = metadata_subset[col].astype(str)  # convert to string to avoid category bug
            if values.nunique() > 0:
                palette = sns.color_palette(
                    "tab20" if "cluster" in col.lower() else "Set2",
                    values.nunique()
                )
                lut = dict(zip(values.unique(), palette))
                lut_maps[col] = lut
                mapped_colors = values.map(lut)
                row_colors_list.append(mapped_colors)

        # Combine all color series safely (no MultiIndex)
        row_colors = pd.concat(row_colors_list, axis=1) if row_colors_list else None

        # ---- Heatmap ----
        g = sns.clustermap(
            df_subset.T,  # rows=genes, cols=cells
            cmap="viridis",
            figsize=(12, 12),
            col_colors=row_colors,
            xticklabels=False,
            yticklabels=True
        )

        # Rotate gene labels
        g.ax_heatmap.set_yticklabels(
            g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10
        )
        g.ax_heatmap.set_xlabel('')
        g.ax_heatmap.set_xticks([])

        # ---- Legends for metadata ----
        for idx, col in enumerate(metadata_subset.columns):
            if col in lut_maps:
                handles = [
                    Patch(facecolor=color, edgecolor='black', label=label)
                    for label, color in lut_maps[col].items()
                ]
                g.ax_col_dendrogram.legend(
                    handles=handles,
                    title=col,
                    bbox_to_anchor=(1.05 + 0.25 * idx, 1),
                    loc='upper left'
                )

        # Save
        plot_path, url = _uniq_plot_path(f"heatmap_{file_id}")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        return {"heatmap_url": url}

    except Exception as e:
        import traceback
        print("🔥 Heatmap generation failed:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# # --- Exploratory Plots + Sankey Endpoint ---

import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import anndata as ad

# ------------------ Exploratory Plots Endpoint ------------------ #
@app.post("/generate_exploratory_plots")
async def generate_exploratory_plots(
    file_id: str = Form(...),
    gene: str = Form("XBP1")
):
    try:
        # Load H5AD
        adata = _load_adata_safe(file_id)
        output_urls = {}

        # ------------------ Ensure UMAP ------------------ #
        if 'X_umap' not in adata.obsm:
            if 'X_pca' not in adata.obsm:
                sc.tl.pca(adata)
            sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_pca')
            sc.tl.umap(adata)

        # ------------------ UMAP Plot ------------------ #
        if gene not in adata.var_names:
            gene = adata.var_names[0]

        umap_path, umap_url = _uniq_plot_path(f"{file_id}_umap.png")
        embedding_data = adata.obsm['X_umap']
        plt.figure(figsize=(7, 6))
        scatter = plt.scatter(
            embedding_data[:, 0], embedding_data[:, 1],
            c=pd.Categorical(adata.obs['cellType']).codes,
            cmap='tab20', s=10
        )
        plt.title(f"UMAP of {gene}")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.colorbar(scatter, ticks=range(len(adata.obs['cellType'].unique())), label='cellType')
        plt.tight_layout()
        plt.savefig(umap_path, dpi=150)
        plt.close()
        output_urls['umap_url'] = umap_url

        # ------------------ Dotplot ------------------ #
        dot_path, dot_url = _uniq_plot_path(f"{file_id}_dotplot.png")
        marker_genes = [g for g in ['EPCAM', 'KRT18', 'CDH1', 'CLDN1', 'MUC1'] if g in adata.var_names]
        if marker_genes:
            sc.pl.dotplot(adata, var_names=marker_genes, groupby='cellType', show=False)
            plt.savefig(dot_path, dpi=150, bbox_inches="tight")
            plt.close()
            output_urls['dotplot_url'] = dot_url

        # ------------------ Stacked Bar ------------------ #
        stack_path, stack_url = _uniq_plot_path(f"{file_id}_stackbar.png")
        obs = adata.obs.copy()
        count_df = obs['cellType'].value_counts().reset_index()
        count_df.columns = ['cellType', 'Count']
        plt.figure(figsize=(8, 5))
        sns.barplot(x='cellType', y='Count', data=count_df, palette="Set2")
        plt.xticks(rotation=45)
        plt.title("Stacked Bar Chart of Cell Types")
        plt.tight_layout()
        plt.savefig(stack_path, dpi=150, bbox_inches="tight")
        plt.close()
        output_urls['stacked_bar_url'] = stack_url

        # ------------------ Sankey Diagram ------------------ #
        try:
            import plotly.graph_objects as go

            # CSV aur H5AD ka path resolve karo (jaise heatmap me)
            csv_path = resolve_csv_path(file_id)
            h5ad_path = _get_adata_path(file_id)

            if not csv_path or not h5ad_path:
                raise FileNotFoundError(f"No CSV or H5AD file found for file_id: {file_id}")

            # Load CSV
            score_df = pd.read_csv(csv_path, index_col=0)

            # Load metadata from h5ad
            cell_info_df = adata.obs[['cellType', 'BC_type']].copy()

            # Merge
            merged_data = pd.merge(
                cell_info_df, score_df.T,
                left_index=True, right_index=True, how='inner'
            )

            # Top metabolites
            avg_scores = merged_data.drop(['cellType', 'BC_type'], axis=1).mean().sort_values(ascending=False)
            top_metabolites = avg_scores.head(5).index.tolist()

            # Nodes
            source_nodes = merged_data['cellType'].unique()
            target_nodes = merged_data['BC_type'].unique()
            all_raw_nodes = list(source_nodes) + top_metabolites + list(target_nodes)
            all_nodes = [f"{i} - {node}" for i, node in enumerate(all_raw_nodes)]
            node_to_index = {node: i for i, node in enumerate(all_raw_nodes)}

            # Links
            links = []
            for cell_type in source_nodes:
                for metabolite in top_metabolites:
                    value = merged_data[merged_data['cellType'] == cell_type][metabolite].mean()
                    if value > 0:
                        links.append({
                            'source': node_to_index[cell_type],
                            'target': node_to_index[metabolite],
                            'value': value
                        })

            for bc_type in target_nodes:
                for metabolite in top_metabolites:
                    value = merged_data[merged_data['BC_type'] == bc_type][metabolite].mean()
                    if value > 0:
                        links.append({
                            'source': node_to_index[metabolite],
                            'target': node_to_index[bc_type],
                            'value': value
                        })

            # Plot Sankey
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15, thickness=20, line=dict(color="black", width=0.5),
                    label=all_nodes
                ),
                link=dict(
                    source=[l['source'] for l in links],
                    target=[l['target'] for l in links],
                    value=[l['value'] for l in links]
                )
            )])
            fig.update_layout(title_text="Sankey Diagram with Multiple Metabolites", font_size=10)

            sankey_path, sankey_url = _uniq_plot_path(f"{file_id}_sankey.png")
            fig.write_image(sankey_path, scale=2)  # kaleido required
            output_urls['sankey_url'] = sankey_url

        except Exception as sankey_err:
            print(f"Sankey diagram failed: {sankey_err}")

        return JSONResponse(output_urls)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Exploratory plots failed: {str(e)}")



# Carbon palettes
color_palettes = {
    "Categorical": [
        "#6929c4", "#1192e8", "#005d5d", "#9f1853", "#fa4d56",
        "#570408", "#198038", "#002d9c", "#ee538b", "#b28600",
        "#009d9a", "#012749", "#8a3800", "#a56eff"
    ],
    "Sequential": [
        "#edf5ff", "#d0e2ff", "#a6c8ff", "#78a9ff", "#4589ff",
        "#0f62fe", "#0043ce", "#002d9c", "#001d6c", "#001141"
    ],
    "Diverging": [
        "#750e13", "#a2191f", "#da1e28", "#fa4d56", "#ff8389",
        "#ffb3b8", "#ffd7d9", "#fff1f1", "#e5f6ff", "#bae6ff",
        "#82cfff", "#33b1ff", "#1192e8", "#0072c3", "#00539a", "#003a6d"
    ]
}


@app.post("/visualize")
async def visualize(
    file_id: str = Form(...),
    embeddings: str = Form(...),
    color_by: str = Form(...),
    palette: str = Form("Normal")   # 👈 new param with default = Normal
):
    try:
        adata = _load_adata_safe(file_id)
        embeddings_list = json.loads(embeddings)

        if not embeddings_list:
            raise HTTPException(status_code=400, detail="No embeddings selected for visualization.")

        results = {}
        for emb_key in embeddings_list:
            if emb_key not in adata.obsm:
                continue

            plot_path, url = _uniq_plot_path(f"visualize_{emb_key}_{file_id}")

            # choose colors
            if palette == "Normal":
                colors = None
            else:
                colors = color_palettes.get(palette, None)

            plot_embedding(
                adata, 
                adata.obsm[emb_key], 
                plot_path, 
                emb_key, 
                color_by, 
                colors=colors
            )
            results[f"{emb_key}_url"] = url

        if not results:
            raise HTTPException(status_code=404, detail="No valid embeddings found to visualize.")

        return results
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")
    

# --- Upload DE Results (curated for UI) ---

@app.post("/upload_de_results")
async def upload_de_results(
    file_id: str = Form(...),
    de_results_file: UploadFile = File(...)
):
    """
    Upload DE results CSV, save curated columns, and store in AnnData.uns
    """
    file_path = _get_adata_path(file_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="H5AD file not found. Upload a project first.")
    
    folder = os.path.dirname(file_path)
    de_results_path = os.path.join(folder, "de_results.csv")

    try:
        # Save uploaded CSV
        with open(de_results_path, "wb") as f:
            f.write(await de_results_file.read())
        
        # Read CSV and keep ONLY UI-relevant columns
        de_results_df = pd.read_csv(de_results_path)
        curated_cols = ['p_value', 'adj_p_value', 'log2FC', 'status']
        de_results_df = de_results_df[[c for c in curated_cols if c in de_results_df.columns]]
        
        # Save to AnnData
        adata = _load_adata_safe(file_id)
        adata.uns['up_down_table'] = json.dumps(de_results_df.to_dict(orient='records'))
        adata.write(file_path)

        return {"message": "DE results uploaded successfully (UI columns saved)"}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process DE results: {str(e)}")


# --- Get DE Results for UI ---
@app.get("/get_up_down_players")
async def get_up_down_players(file_id: str):
    """
    Return top DE results (curated columns) for UI
    """
    file_path = _get_adata_path(file_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="H5AD file not found. Upload a project first.")
    
    try:
        adata = _load_adata_safe(file_id)
        if 'up_down_table' not in adata.uns:
            return {"file_id": file_id, "up_down_players": []}
        
        de_data = json.loads(adata.uns['up_down_table'])
        return {"file_id": file_id, "up_down_players": de_data}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to retrieve DE results: {str(e)}")



@app.get("/download_project_technologies")
async def download_technologies():
    tech_file_path = os.path.join("static", "downloads", "used_technologies.txt")
    os.makedirs(os.path.dirname(tech_file_path), exist_ok=True)
    with open(tech_file_path, "w", encoding="utf-8") as f:
        f.write("🔧 Technologies Used in This Project\n")
        f.write("-----------------------------------\n")
        f.write("• Python 3.10\n")
        f.write("• FastAPI\n")
        f.write("• HTML5, CSS3, Vanilla JS (frontend)\n")
        f.write("• Scanpy & AnnData\n")
        f.write("• NumPy, Pandas\n")
        f.write("• Matplotlib, Seaborn\n")
        f.write("• scikit-learn (PCA)\n")
        f.write("• UMAP-learn\n")
        f.write("• Jinja2 Templates\n")
    return FileResponse(tech_file_path, media_type='text/plain', filename="used_technologies.txt")
 
@app.get("/get_available_embeddings")
def get_available_embeddings(file_id: str):
    try:
        adata = _load_adata_safe(file_id)
        obsm_keys = list(adata.obsm.keys())
        
        curated_keys = []
        priority_keys = ['X_hgraph_pca', 'X_hgraph_umap', 'X_std_pca', 'X_std_umap', 'X_pca', 'X_umap']
        for key in priority_keys:
            if key in obsm_keys:
                curated_keys.append(key)
        
        other_keys = [k for k in obsm_keys if k not in curated_keys]
        curated_keys.extend(other_keys[:7 - len(curated_keys)])
        
        return {"obsm_keys": curated_keys}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found for id: {file_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_obs_columns")
def get_obs_columns(file_id: str):
    try:
        adata = _load_adata_safe(file_id)
        obs_keys = list(adata.obs.columns)

        priority_keys = [
            "cluster", "celltype", "batch",
            "condition", "treatment", "timepoint"
        ]

        curated_keys = [k for k in priority_keys if k in obs_keys]
        other_keys = [k for k in obs_keys if k not in curated_keys]
        curated_keys.extend(other_keys[: max(0, 7 - len(curated_keys))])

        return {"obs_keys": curated_keys}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found for id: {file_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_de_results")
async def download_de_results(file_id: str):
    folder = os.path.join(UPLOAD_FOLDER, file_id)
    de_csv_path = os.path.join(folder, "de_results.csv")
    if not os.path.exists(de_csv_path):
        raise HTTPException(status_code=404, detail="DE results CSV not found")
    return FileResponse(de_csv_path, media_type="text/csv", filename=f"de_results_{file_id}.csv")

@app.get("/generate_full_report_pdf")
def generate_full_report_pdf(file_id: str):
    pdf_path = os.path.join(REPORTS_FOLDER, f"full_report_{file_id}.pdf")
    
    try:
        # Always generate latest PDF
        generate_full_report(file_id, pdf_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project data not found for ID: {file_id}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")
    
    return FileResponse(
        pdf_path,   
        media_type='application/pdf',
        filename=f"full_report_{file_id}.pdf"
    )

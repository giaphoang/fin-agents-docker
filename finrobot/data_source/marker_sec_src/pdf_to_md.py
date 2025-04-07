# import subprocess
import os

# Create dummy implementations for missing modules
try:
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
    from marker.output import save_markdown
    
    # Try to import segformer from surya
    try:
        from surya.model.detection import segformer
    except ImportError:
        print("Warning: Cannot import segformer from surya.model.detection. This may affect functionality.")
        segformer = None
    
    MARKER_AVAILABLE = True
except ImportError:
    print("Warning: marker modules not found. PDF to Markdown conversion will not be available.")
    # Define dummy functions to prevent import errors
    def convert_single_pdf(*args, **kwargs):
        raise NotImplementedError("marker package not installed. PDF to Markdown conversion not available.")
    
    def load_all_models(*args, **kwargs):
        raise NotImplementedError("marker package not installed. PDF to Markdown conversion not available.")
    
    def save_markdown(*args, **kwargs):
        raise NotImplementedError("marker package not installed. PDF to Markdown conversion not available.")
    
    segformer = None
    MARKER_AVAILABLE = False

SAVE_DIR = "output/SEC_EDGAR_FILINGS_MD"


# def run_marker(input_ticker_year_path:str,ticker:str,year:str,workers:int=4,max_workers:int=8,num_chunks:int=1):
def run_marker(
    input_ticker_year_path: str, output_ticker_year_path:str,batch_multiplier: int = 2
):
    if not MARKER_AVAILABLE:
        print("Error: marker package not installed. PDF to Markdown conversion not available.")
        print("Please install marker using: pip install git+https://github.com/VikParuchuri/marker.git")
        print("And surya using: pip install git+https://github.com/VikParuchuri/surya.git")
        return

    # subprocess.run(["marker", input_ticker_year_path,output_ticker_year_path,  "--workers", str(workers), "--num_chunks",str(num_chunks),"--max", str(max_workers) ,"--metadata_file", path_to_metadata])
    # return
    model_lst = load_all_models()
    for input_path in os.listdir(input_ticker_year_path):
        if not input_path.endswith(".pdf"):
            continue
        input_path = os.path.join(input_ticker_year_path, input_path)
        full_text, images, out_meta = convert_single_pdf(
            input_path, model_lst, langs=["English"], batch_multiplier=batch_multiplier
        )
        fname = os.path.basename(input_path)
        subfolder_path = save_markdown(
            output_ticker_year_path, fname, full_text, images, out_meta
        )
        print(f"Saved markdown to the {subfolder_path} folder")
    del model_lst
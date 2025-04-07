try:
    from finrobot.data_source.marker_sec_src.sec_filings_to_pdf import sec_save_pdfs
except ImportError:
    print("Warning: Import error with sec_filings_to_pdf. PDF saving functionality may not be available.")
    # Provide a dummy implementation
    def sec_save_pdfs(*args, **kwargs):
        print("PDF saving functionality not available due to missing dependencies.")
        return None

try:
    from finrobot.data_source.marker_sec_src.pdf_to_md import run_marker
except ImportError:
    print("Warning: Import error with pdf_to_md. PDF to Markdown conversion may not be available.")
    # Provide a dummy implementation
    def run_marker(*args, **kwargs):
        print("PDF to Markdown conversion not available due to missing dependencies.")
        return None

try:
    from finrobot.data_source.marker_sec_src.pdf_to_md_parallel import run_marker_mp
except ImportError:
    print("Warning: Import error with pdf_to_md_parallel. PDF to Markdown parallel conversion may not be available.")
    # Provide a dummy implementation
    def run_marker_mp(*args, **kwargs):
        print("PDF to Markdown parallel conversion not available due to missing dependencies.")
        return None

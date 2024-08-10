from io import BytesIO
import pathlib
import tempfile
from urllib.parse import parse_qs, urlparse, urlencode
from zipfile import ZipFile
import streamlit as st
import platform
import requests
import photoacoustic

st.set_page_config(
   page_title="Photoacoustic Analysis",
   page_icon="🧊",
)

st.title("Photoacoustic Analysis 💡🎤")
st.markdown("""
1. Save your data using the following naming [convention](https://docs.google.com/document/d/1JBwmuLxCsvaFCDBpr_KElQ-tdxVQ7ZsdQhIBlSiCvc0/pub).
2. Copy your data to sciebo in a folder that has been shared to with `Download / View / Edit` permissions.
3. In your browser, navigate to your newly uploaded data and copy the URL.
4. Paste the URL into the form below.
5. Click on Analyze
""")


def build_download_link(url: str) -> str:
    pu = urlparse(url)
    params = parse_qs(pu.query)
    params_path = params.get("path", [""])[0]
    if "/" in params_path:
        params["path"], params["files"] = params_path.rsplit("/", 1)
        pu = pu._replace(query=urlencode(params))
    pu = pu._replace(path=pu.path + "/download")
    return pu.geturl()

def get_key_folder(url:str) -> tuple[str, str]:
    pu = urlparse(url)
    params = parse_qs(pu.query)

    return pu.path.split("/")[-1], params.get("path", [""])[-1]


class Progress:

    def __init__(self, content: dict[str, int], folder_bar, file_bar):
        self.content = content
        self.folder_bar = folder_bar
        self.file_bar = file_bar
        self.progress = {}

    def __call__(self, s: str):
        folder, file = s.split("/", 1)
        if folder not in self.progress:
            self.progress[folder] = []
            self.folder_bar.progress(len(self.progress)/len(self.content), text=folder)

        files = self.progress[folder]
        cnt = self.content.get(folder, 100)
        if not file in files:
            files.append(file)
            self.file_bar.progress(len(files)/cnt, text=file)



with st.form("Source data"):
    url_val = st.text_input("sciebo URL")
    plot_repetitions = st.checkbox("Plot timetrace for repeats", value=True)

    # Every form must have a submit button.
    submitted = st.form_submit_button("🚀 Analyze")
    if submitted:
        url_val = url_val.strip()
        url = build_download_link(url_val)

        with st.spinner(f'Downloading {url}'):
            response = requests.get(url)

        if response.status_code != 200:
            st.error(f"Cannot download data from sciebo (status code {response.status_code})")
        else:
            with tempfile.TemporaryDirectory() as folder:
                folder = pathlib.Path(folder)
                st.info(f"💾 Downloaded {len(response.content)/1024:.2f} Mb from sciebo")
                data = ZipFile(BytesIO(response.content)).extractall(folder)
                for p in folder.iterdir():
                    content = {
                        sp.stem: len(list(sp.glob('*.txt'))) for sp in p.iterdir() 
                        if sp.is_dir() and not sp.stem.startswith("_")
                    }
                    st.info(f"📂 Data content of {p.stem}  \n  \n" + "  \n".join([f"{k}: {v} txt files" for k, v in content.items()]))                                            
                    with st.spinner("Analyzing ..."):
                        folder_bar = st.progress(0, "Folders")
                        file_bar = st.progress(0, "Files")
                        progress = Progress(content, folder_bar, file_bar)
                        photoacoustic.analyze(
                            p, {
                                "on_progress": progress, 
                                "on_error": st.error,
                                "plot_time_trace_rep": plot_repetitions
                                }
                            )
                    folder_bar.empty()
                    file_bar.empty()
                    st.info("👍 Analysis done!")

                headers = {'X-Requested-With': 'XMLHttpRequest',}
                key, folder = get_key_folder(url_val)
                upload_ok = True
                with st.spinner("Uploading results to sciebo ..."):
                    for ext in ("xlsx", "pdf"):                
                        response = requests.put(
                            f'https://uni-muenster.sciebo.de/public.php/webdav/{folder}/summary.{ext}',
                            headers=headers,
                            data=(p / f"summary.{ext}").read_bytes(),
                            verify=False,
                            auth=(key, ''),
                        )
                        if 200 <= response.status_code < 300:
                            st.info(f"💾 summary.{ext} stored in sciebo")
                        else:
                            upload_ok = False
                            st.error(f"Problem found while uploading summary.{ext} (status code {response.status_code})")

                if upload_ok:
                    st.info(f"🎉 Process complete!  \n  \nReload your sciebo page to get your results or follow this link {url_val}")    
                else:
                    st.warn(f"Process complete but problems found while uploading results  \nReload your sciebo page to check if your results are present or follow this link {url_val}")    


with st.expander("Package versions"):
    versions = [
        ("Python", platform.python_version()),
        ("requests", requests.__version__)
    ] + photoacoustic.versions()
    st.text("  \n".join([f"{k}: {v}" for k, v in versions]))
        
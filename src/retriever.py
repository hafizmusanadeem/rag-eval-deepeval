import os
import re
import glob
import time

import httpx
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()  # loads OPENROUTER_API_KEY from .env

DATA_DIR = "data"
DB_DIR = "chroma_store"


# 0. EMBEDDINGS ---- talks to OpenRouter directly via httpx, bypassing the
#    openai SDK entirely (no tiktoken pre-tokenization, no auto-injected
#    OpenAI-Organization/OpenAI-Project headers that were causing the 401s)
class OpenRouterEmbeddings(Embeddings):

    def __init__(
        self,
        model,
        api_key,
        base_url="https://openrouter.ai/api/v1"
    ):
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is missing or empty"
            )

        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _post(self, input_list):

        max_retries = 3

        for attempt in range(max_retries):

            try:

                r = httpx.post(
                    f"{self.base_url}/embeddings",

                    headers={
                        "Authorization": f"Bearer {self.api_key}"
                    },

                    json={
                        "model": self.model,
                        "input": input_list
                    },

                    timeout=120,
                )

                if r.status_code != 200:
                    raise RuntimeError(
                        f"OpenRouter embeddings error "
                        f"{r.status_code}: {r.text}"
                    )

                data = r.json()["data"]

                return [
                    item["embedding"]
                    for item in data
                ]

            except (
                httpx.RemoteProtocolError,
                httpx.ReadTimeout,
                httpx.ConnectError,
            ) as e:

                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Embedding request failed after "
                        f"{max_retries} attempts: {e}"
                    ) from e

                wait_time = 2 ** attempt

                print(
                    f"Temporary HTTP error: "
                    f"{type(e).__name__}. "
                    f"Retrying in {wait_time}s "
                    f"(attempt {attempt + 2}/{max_retries})..."
                )

                time.sleep(wait_time)

    def embed_documents(self, texts):

        batch_size = 50
        embeddings = []

        total_batches = (
            len(texts) + batch_size - 1
        ) // batch_size

        for i in range(0, len(texts), batch_size):

            batch = texts[i:i + batch_size]

            batch_number = i // batch_size + 1

            print(
                f"Embedding batch "
                f"{batch_number}/{total_batches}: "
                f"{len(batch)} texts"
            )

            batch_embeddings = self._post(batch)

            embeddings.extend(batch_embeddings)

        return embeddings

    def embed_query(self, text):
        return self._post([text])[0]

# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():

    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store():
    embeddings = OpenRouterEmbeddings(
        model="liquid/lfm-2.5-embedding-350m:free",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    docs = load_transcripts()

    if not docs:
        raise FileNotFoundError(
            f"No .vtt files found in '{DATA_DIR}/'. "
            f"Check that DATA_DIR is correct relative to your current working directory."
        )

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=70,
    ).split_documents(docs)

    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


# 3. TRY IT ---- python src/retriever.py
if __name__ == "__main__":

    retriever = build_retriever()


    results = retriever.invoke("what is regression testing?")

    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")
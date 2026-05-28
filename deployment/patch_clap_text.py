"""
Patch for AudioMuse-AI tasks/clap_analyzer.py
Replaces DCLAP RoBERTa text encoder with GTE + projection for multilingual support.

Usage:
    docker cp gte_to_dclap_proj.onnx audiomuse-ai-flask-app:/app/model/
    docker cp patch_clap_text.py audiomuse-ai-flask-app:/tmp/
    docker exec audiomuse-ai-flask-app python3 /tmp/patch_clap_text.py
    docker exec audiomuse-ai-flask-app supervisorctl -c /etc/supervisor/conf.d/supervisord.conf restart flask
"""
import re

TARGET = '/app/tasks/clap_analyzer.py'

with open(TARGET, 'r') as f:
    content = f.read()

insertion_point = "_label_text_embeddings_cache = None"
projection_globals = """_label_text_embeddings_cache = None
_gte_projection_session = None
_GTE_PROJECTION_PATH = '/app/model/gte_to_dclap_proj.onnx'"""

content = content.replace(insertion_point, projection_globals, 1)

old_func_pattern = r'def get_text_embedding\(query_text: str\).*?(?=\ndef get_text_embeddings_batch)'

new_func = '''def get_text_embedding(query_text: str) -> Optional[np.ndarray]:
    """
    Get CLAP-compatible 512-dim embedding for a text query.
    Uses GTE + projection if available, falls back to DCLAP RoBERTa.
    """
    global _gte_projection_session

    if not config.CLAP_ENABLED:
        return None

    if os.path.exists(_GTE_PROJECTION_PATH):
        try:
            from lyrics.gte_onnx import embed_text, load_gte_model
            load_gte_model()
            gte_emb = embed_text(query_text)
            if gte_emb is None:
                return _get_text_embedding_dclap(query_text)

            if _gte_projection_session is None:
                import onnxruntime as ort
                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_opts.intra_op_num_threads = 1
                sess_opts.inter_op_num_threads = 1
                _gte_projection_session = ort.InferenceSession(
                    _GTE_PROJECTION_PATH, sess_opts,
                    providers=['CPUExecutionProvider'])
                logger.info("GTE-to-DCLAP projection loaded (multilingual text search)")

            proj_input = gte_emb.reshape(1, -1).astype(np.float32)
            projected = _gte_projection_session.run(None, {'input': proj_input})[0][0]
            projected = projected / (np.linalg.norm(projected) + 1e-9)
            return projected.astype(np.float32)

        except Exception as e:
            logger.warning(f"GTE+projection failed ({e}), falling back to DCLAP RoBERTa")
            return _get_text_embedding_dclap(query_text)
    else:
        return _get_text_embedding_dclap(query_text)


def _get_text_embedding_dclap(query_text: str) -> Optional[np.ndarray]:
    """Original DCLAP RoBERTa text embedding (fallback)."""
    try:
        session = get_clap_text_model()
        tokenizer = get_tokenizer()

        encoded = tokenizer(
            query_text,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )

        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)

        onnx_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        outputs = session.run(None, onnx_inputs)
        text_embedding = outputs[0][0]
        text_embedding = text_embedding / np.linalg.norm(text_embedding)
        return text_embedding

    except Exception as e:
        logger.error(f"Failed to get text embedding for '{query_text}': {e}")
        import traceback
        traceback.print_exc()
        return None


'''

content = re.sub(old_func_pattern, new_func, content, flags=re.DOTALL)

old_is_loaded = "def is_clap_text_loaded():\n    \"\"\"Check if CLAP text model is currently loaded.\"\"\"\n    return _text_session is not None"
new_is_loaded = "def is_clap_text_loaded():\n    \"\"\"Check if CLAP text model is currently loaded.\"\"\"\n    return _text_session is not None or _gte_projection_session is not None"
content = content.replace(old_is_loaded, new_is_loaded)

old_unload_line = "        _tokenizer = None\n        _cached_dummy_input_ids = None"
new_unload_line = "        _tokenizer = None\n        _cached_dummy_input_ids = None\n        global _gte_projection_session\n        _gte_projection_session = None"
content = content.replace(old_unload_line, new_unload_line, 1)

with open(TARGET, 'w') as f:
    f.write(content)

print("Patch applied successfully")

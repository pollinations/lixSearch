from .retrievalSystem import RetrievalSystem


def initialize_retrieval_system() -> RetrievalSystem:
    global _retrieval_system
    _retrieval_system = RetrievalSystem.get_instance()
    return _retrieval_system


def get_retrieval_system() -> RetrievalSystem:
    if _retrieval_system is None:
        return initialize_retrieval_system()
    return _retrieval_system
